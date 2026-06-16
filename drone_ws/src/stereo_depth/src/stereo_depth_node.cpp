#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <boost/bind.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

namespace
{
using Image = sensor_msgs::Image;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

struct PairConfig {
    std::string left_topic;
    std::string right_topic;
    std::string points_topic;
    tf2::Vector3 camera_offset_body_frd;
    bool forward = true;
    double disparity_sign = 1.0;
    ros::Time last_publish_time;
};

cv::Mat imageToBgr(const sensor_msgs::ImageConstPtr& msg)
{
    const cv_bridge::CvImageConstPtr image =
        cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    return image->image;
}

cv::Mat bgrToGray(const cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

int normalizeNumDisparities(int value)
{
    value = std::max(16, value);
    return ((value + 15) / 16) * 16;
}

int normalizeBlockSize(int value)
{
    value = std::max(3, value);
    if (value % 2 == 0) {
        ++value;
    }
    return value;
}

geometry_msgs::Point nedToWorld(double x, double y, double z)
{
    geometry_msgs::Point p;
    p.x = x;
    p.y = -y;
    p.z = -z;
    return p;
}

tf2::Vector3 opticalPointToBodyFrd(
    const tf2::Vector3& optical_point,
    const tf2::Vector3& camera_offset_body_frd,
    bool forward)
{
    const tf2::Vector3 point_link(
        optical_point.z(),
        optical_point.x(),
        optical_point.y());

    if (forward) {
        return camera_offset_body_frd + point_link;
    }

    return camera_offset_body_frd + tf2::Vector3(
        -point_link.x(),
        -point_link.y(),
        point_link.z());
}

class StereoDepthNode {
public:
    StereoDepthNode()
        : nh_()
        , pnh_("~")
    {
        pnh_.param("width", width_, 960);
        pnh_.param("height", height_, 720);
        pnh_.param("horizontal_fov_deg", horizontal_fov_deg_, 60.0);
        pnh_.param("baseline_m", baseline_m_, 0.30);
        pnh_.param("min_depth_m", min_depth_m_, 0.4);
        pnh_.param("max_depth_m", max_depth_m_, 40.0);
        pnh_.param("sample_step", sample_step_, 4);
        pnh_.param("num_disparities", num_disparities_, 96);
        pnh_.param("block_size", block_size_, 7);
        pnh_.param("process_scale", process_scale_, 0.5);
        pnh_.param("max_rate_hz", max_rate_hz_, 3.0);
        pnh_.param("sync_queue_size", sync_queue_size_, 1);
        pnh_.param<std::string>("odom_topic", odom_topic_, "/eskf_odom");
        pnh_.param<std::string>("world_frame_id", world_frame_id_, "world");

        sample_step_ = std::max(1, sample_step_);
        num_disparities_ = normalizeNumDisparities(num_disparities_);
        block_size_ = normalizeBlockSize(block_size_);
        process_scale_ = std::min(1.0, std::max(0.1, process_scale_));
        sync_queue_size_ = std::max(1, sync_queue_size_);
        fx_ = (static_cast<double>(width_) * 0.5) /
              std::tan(horizontal_fov_deg_ * M_PI / 360.0);
        fy_ = fx_;
        cx_ = static_cast<double>(width_) * 0.5;
        cy_ = static_cast<double>(height_) * 0.5;

        PairConfig front;
        pnh_.param<std::string>("front_left_topic", front.left_topic, "/airsim_node/drone_1/front_left/Scene");
        pnh_.param<std::string>("front_right_topic", front.right_topic, "/airsim_node/drone_1/front_right/Scene");
        pnh_.param<std::string>("front_points_topic", front.points_topic, "/stereo_depth/front/points");
        pnh_.param("front_disparity_sign", front.disparity_sign, 1.0);
        front.camera_offset_body_frd = tf2::Vector3(0.175, -0.15, 0.0);
        front.forward = true;

        PairConfig back;
        pnh_.param<std::string>("back_left_topic", back.left_topic, "/airsim_node/drone_1/back_left/Scene");
        pnh_.param<std::string>("back_right_topic", back.right_topic, "/airsim_node/drone_1/back_right/Scene");
        pnh_.param<std::string>("back_points_topic", back.points_topic, "/stereo_depth/back/points");
        pnh_.param("back_disparity_sign", back.disparity_sign, 1.0);
        back.camera_offset_body_frd = tf2::Vector3(-0.175, -0.15, 0.0);
        back.forward = false;

        front_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(front.points_topic, 1);
        back_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(back.points_topic, 1);
        odom_sub_ = nh_.subscribe(odom_topic_, 1, &StereoDepthNode::odomCallback, this);

        front_left_sub_.reset(new message_filters::Subscriber<Image>(nh_, front.left_topic, 1));
        front_right_sub_.reset(new message_filters::Subscriber<Image>(nh_, front.right_topic, 1));
        back_left_sub_.reset(new message_filters::Subscriber<Image>(nh_, back.left_topic, 1));
        back_right_sub_.reset(new message_filters::Subscriber<Image>(nh_, back.right_topic, 1));

        front_sync_.reset(new Synchronizer(SyncPolicy(sync_queue_size_), *front_left_sub_, *front_right_sub_));
        back_sync_.reset(new Synchronizer(SyncPolicy(sync_queue_size_), *back_left_sub_, *back_right_sub_));
        front_sync_->registerCallback(boost::bind(&StereoDepthNode::frontCallback, this, _1, _2));
        back_sync_->registerCallback(boost::bind(&StereoDepthNode::backCallback, this, _1, _2));

        front_config_ = front;
        back_config_ = back;
        front_matcher_ = createMatcher();
        back_matcher_ = createMatcher();

        ROS_INFO("stereo_depth: fx %.2f baseline %.2f m scale %.2f max %.1f Hz, front %s -> %s, back %s -> %s",
                 fx_,
                 baseline_m_,
                 process_scale_,
                 max_rate_hz_,
                 front.left_topic.c_str(),
                 front.points_topic.c_str(),
                 back.left_topic.c_str(),
                 back.points_topic.c_str());
    }

private:
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        latest_odom_ = *msg;
        has_odom_ = true;
    }

    void frontCallback(const Image::ConstPtr& left, const Image::ConstPtr& right)
    {
        publishCloud(left, right, front_config_, front_pub_, front_matcher_);
    }

    void backCallback(const Image::ConstPtr& left, const Image::ConstPtr& right)
    {
        publishCloud(left, right, back_config_, back_pub_, back_matcher_);
    }

    cv::Ptr<cv::StereoSGBM> createMatcher() const
    {
        cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(
            0,
            num_disparities_,
            block_size_,
            8 * block_size_ * block_size_,
            32 * block_size_ * block_size_,
            1,
            63,
            10,
            100,
            32,
            cv::StereoSGBM::MODE_SGBM_3WAY);
        return matcher;
    }

    void publishCloud(
        const Image::ConstPtr& left_msg,
        const Image::ConstPtr& right_msg,
        PairConfig& config,
        const ros::Publisher& publisher,
        const cv::Ptr<cv::StereoSGBM>& matcher)
    {
        if (!has_odom_) {
            ROS_WARN_THROTTLE(1.0, "stereo_depth: waiting for odometry on %s", odom_topic_.c_str());
            return;
        }

        if (publisher.getNumSubscribers() == 0) {
            return;
        }

        tf2::Quaternion body_q(
            latest_odom_.pose.pose.orientation.x,
            latest_odom_.pose.pose.orientation.y,
            latest_odom_.pose.pose.orientation.z,
            latest_odom_.pose.pose.orientation.w);
        if (body_q.length2() < 1e-12) {
            return;
        }
        body_q.normalize();

        const tf2::Vector3 body_pos_ned(
            latest_odom_.pose.pose.position.x,
            latest_odom_.pose.pose.position.y,
            latest_odom_.pose.pose.position.z);

        const ros::Time now = ros::Time::now();
        if (max_rate_hz_ > 0.0 &&
            !config.last_publish_time.isZero() &&
            (now - config.last_publish_time).toSec() < 1.0 / max_rate_hz_) {
            return;
        }
        config.last_publish_time = now;

        cv::Mat left_bgr;
        cv::Mat right_bgr;
        try {
            left_bgr = imageToBgr(left_msg);
            right_bgr = imageToBgr(right_msg);
        } catch (const cv_bridge::Exception& e) {
            ROS_WARN_THROTTLE(1.0, "stereo_depth: cv_bridge failed: %s", e.what());
            return;
        }

        if (left_bgr.empty() || right_bgr.empty() || left_bgr.size() != right_bgr.size()) {
            ROS_WARN_THROTTLE(1.0, "stereo_depth: invalid stereo image sizes");
            return;
        }

        cv::Mat left_proc = left_bgr;
        cv::Mat right_proc = right_bgr;
        if (process_scale_ < 0.999) {
            cv::resize(left_bgr, left_proc, cv::Size(), process_scale_, process_scale_, cv::INTER_AREA);
            cv::resize(right_bgr, right_proc, cv::Size(), process_scale_, process_scale_, cv::INTER_AREA);
        }

        cv::Mat left_gray = bgrToGray(left_proc);
        cv::Mat right_gray = bgrToGray(right_proc);
        cv::Mat disparity_raw;
        matcher->compute(left_gray, right_gray, disparity_raw);

        const double scaled_fx = fx_ * process_scale_;
        const double scaled_fy = fy_ * process_scale_;
        const double scaled_cx = cx_ * process_scale_;
        const double scaled_cy = cy_ * process_scale_;

        const int max_points =
            ((left_proc.rows + sample_step_ - 1) / sample_step_) *
            ((left_proc.cols + sample_step_ - 1) / sample_step_);

        sensor_msgs::PointCloud2 cloud;
        cloud.header.stamp = left_msg->header.stamp.isZero() ? ros::Time::now() : left_msg->header.stamp;
        cloud.header.frame_id = world_frame_id_;
        cloud.height = 1;
        cloud.is_dense = false;

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(max_points);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud, "b");

        int count = 0;
        for (int v = 0; v < left_proc.rows; v += sample_step_) {
            const auto* disp_row = disparity_raw.ptr<int16_t>(v);
            const auto* color_row = left_proc.ptr<cv::Vec3b>(v);
            for (int u = 0; u < left_proc.cols; u += sample_step_) {
                const double disparity = config.disparity_sign * static_cast<double>(disp_row[u]) / 16.0;
                if (disparity <= 0.1) {
                    continue;
                }

                const double depth = scaled_fx * baseline_m_ / disparity;
                if (!std::isfinite(depth) || depth < min_depth_m_ || depth > max_depth_m_) {
                    continue;
                }

                const tf2::Vector3 point_optical(
                    (static_cast<double>(u) - scaled_cx) * depth / scaled_fx,
                    (static_cast<double>(v) - scaled_cy) * depth / scaled_fy,
                    depth);
                const tf2::Vector3 point_body =
                    opticalPointToBodyFrd(point_optical, config.camera_offset_body_frd, config.forward);
                const tf2::Vector3 point_ned = body_pos_ned + tf2::quatRotate(body_q, point_body);
                const geometry_msgs::Point point_world = nedToWorld(
                    point_ned.x(),
                    point_ned.y(),
                    point_ned.z());

                *iter_x = static_cast<float>(point_world.x);
                *iter_y = static_cast<float>(point_world.y);
                *iter_z = static_cast<float>(point_world.z);

                const cv::Vec3b& color = color_row[u];
                *iter_b = color[0];
                *iter_g = color[1];
                *iter_r = color[2];

                ++iter_x;
                ++iter_y;
                ++iter_z;
                ++iter_r;
                ++iter_g;
                ++iter_b;
                ++count;
            }
        }

        modifier.resize(count);
        publisher.publish(cloud);
        ROS_DEBUG_THROTTLE(1.0, "stereo_depth: published %d world points", count);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher front_pub_;
    ros::Publisher back_pub_;
    ros::Subscriber odom_sub_;

    std::unique_ptr<message_filters::Subscriber<Image>> front_left_sub_;
    std::unique_ptr<message_filters::Subscriber<Image>> front_right_sub_;
    std::unique_ptr<message_filters::Subscriber<Image>> back_left_sub_;
    std::unique_ptr<message_filters::Subscriber<Image>> back_right_sub_;
    std::unique_ptr<Synchronizer> front_sync_;
    std::unique_ptr<Synchronizer> back_sync_;
    cv::Ptr<cv::StereoSGBM> front_matcher_;
    cv::Ptr<cv::StereoSGBM> back_matcher_;

    PairConfig front_config_;
    PairConfig back_config_;

    int width_ = 960;
    int height_ = 720;
    int sample_step_ = 4;
    int num_disparities_ = 96;
    int block_size_ = 7;
    int sync_queue_size_ = 1;
    double horizontal_fov_deg_ = 60.0;
    double baseline_m_ = 0.30;
    double min_depth_m_ = 0.4;
    double max_depth_m_ = 40.0;
    double process_scale_ = 0.5;
    double max_rate_hz_ = 3.0;
    double fx_ = 831.0;
    double fy_ = 831.0;
    double cx_ = 480.0;
    double cy_ = 360.0;
    std::string odom_topic_ = "/eskf_odom";
    std::string world_frame_id_ = "world";
    nav_msgs::Odometry latest_odom_;
    bool has_odom_ = false;
};
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stereo_depth_node");
    StereoDepthNode node;
    ros::spin();
    return 0;
}
