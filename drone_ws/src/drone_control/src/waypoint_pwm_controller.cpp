#include <ros/ros.h>

#include <Eigen/Dense>
#include <airsim_ros/RotorPWM.h>
#include <airsim_ros/Takeoff.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "drone_control/PDcontroller.hpp"

namespace
{
constexpr double kMinQuatNorm = 1e-12;

Eigen::Vector3d nedToWorld(const Eigen::Vector3d& ned)
{
    return Eigen::Vector3d(ned.x(), -ned.y(), -ned.z());
}

Eigen::Matrix3d nedToWorldMatrix()
{
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
    m(1, 1) = -1.0;
    m(2, 2) = -1.0;
    return m;
}

double clampValue(double value, double min_value, double max_value)
{
    return std::max(min_value, std::min(max_value, value));
}

double yawFromRotation(const Eigen::Matrix3d& rotation)
{
    tf2::Matrix3x3 tf_rotation(
        rotation(0, 0), rotation(0, 1), rotation(0, 2),
        rotation(1, 0), rotation(1, 1), rotation(1, 2),
        rotation(2, 0), rotation(2, 1), rotation(2, 2));
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    tf_rotation.getRPY(roll, pitch, yaw);
    return yaw;
}

class WaypointPwmController
{
public:
    WaypointPwmController()
        : nh_()
        , pnh_("~")
    {
        pnh_.param<std::string>("odom_topic", odom_topic_, "/eskf_odom");
        pnh_.param<std::string>("target_topic", target_topic_, "/my_drone/current_waypoint");
        pnh_.param<std::string>("pwm_topic", pwm_topic_, "/airsim_node/drone_1/rotor_pwm_cmd");
        pnh_.param<std::string>("takeoff_service", takeoff_service_, "/airsim_node/drone_1/takeoff");
        pnh_.param("auto_takeoff", auto_takeoff_, true);
        pnh_.param("control_rate_hz", control_rate_hz_, 200.0);
        pnh_.param("max_target_age_s", max_target_age_s_, 2.0);
        pnh_.param("post_takeoff_wait_s", post_takeoff_wait_s_, 3.0);
        pnh_.param("max_pwm", max_pwm_, 1.0);
        pnh_.param("min_pwm", min_pwm_, 0.0);

        pwm_pub_ = nh_.advertise<airsim_ros::RotorPWM>(pwm_topic_, 1);
        odom_sub_ = nh_.subscribe(odom_topic_, 1, &WaypointPwmController::odomCallback, this);
        target_sub_ = nh_.subscribe(target_topic_, 1, &WaypointPwmController::targetCallback, this);

        if (auto_takeoff_) {
            takeoff_client_ = nh_.serviceClient<airsim_ros::Takeoff>(takeoff_service_);
            ROS_INFO("waypoint_pwm_controller: waiting for takeoff service %s", takeoff_service_.c_str());
            takeoff_client_.waitForExistence();
            airsim_ros::Takeoff srv;
            srv.request.waitOnLastTask = 0;
            if (takeoff_client_.call(srv)) {
                ROS_INFO("waypoint_pwm_controller: takeoff requested");
                if (post_takeoff_wait_s_ > 0.0) {
                    ROS_INFO("waypoint_pwm_controller: waiting %.1fs after takeoff", post_takeoff_wait_s_);
                    ros::Duration(post_takeoff_wait_s_).sleep();
                }
            } else {
                ROS_WARN("waypoint_pwm_controller: takeoff service call failed");
            }
        }

        ROS_INFO(
            "waypoint_pwm_controller: odom=%s target=%s pwm=%s rate=%.1f",
            odom_topic_.c_str(),
            target_topic_.c_str(),
            pwm_topic_.c_str(),
            control_rate_hz_);
    }

    void run()
    {
        ros::Rate rate(control_rate_hz_);
        while (ros::ok()) {
            ros::spinOnce();
            publishControl();
            rate.sleep();
        }
    }

private:
    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        latest_target_ = *msg;
        latest_target_time_ = ros::Time::now();
        has_target_ = true;
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        latest_odom_ = *msg;
        has_odom_ = true;
    }

    bool targetFresh() const
    {
        if (!has_target_) {
            return false;
        }
        return max_target_age_s_ <= 0.0 ||
               (ros::Time::now() - latest_target_time_).toSec() <= max_target_age_s_;
    }

    bool buildState(Eigen::VectorXf& x_real, double& current_yaw)
    {
        const auto& pose = latest_odom_.pose.pose;
        Eigen::Quaterniond q_ned_body(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);
        if (q_ned_body.norm() < kMinQuatNorm) {
            return false;
        }
        q_ned_body.normalize();

        const Eigen::Vector3d position_ned(
            pose.position.x,
            pose.position.y,
            pose.position.z);
        const Eigen::Vector3d position_world = nedToWorld(position_ned);

        const Eigen::Matrix3d r_ned_body = q_ned_body.toRotationMatrix();
        const Eigen::Matrix3d s = nedToWorldMatrix();
        const Eigen::Matrix3d r_world_body_flu = s * r_ned_body * s;

        const Eigen::Vector3d velocity_ned(
            latest_odom_.twist.twist.linear.x,
            latest_odom_.twist.twist.linear.y,
            latest_odom_.twist.twist.linear.z);
        const Eigen::Vector3d velocity_body_ned = r_ned_body.transpose() * velocity_ned;
        const Eigen::Vector3d velocity_body_flu = s * velocity_body_ned;

        const Eigen::Vector3d angular_ned(
            latest_odom_.twist.twist.angular.x,
            latest_odom_.twist.twist.angular.y,
            latest_odom_.twist.twist.angular.z);
        const Eigen::Vector3d angular_flu = s * angular_ned;

        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        tf2::Matrix3x3 tf_rotation(
            r_world_body_flu(0, 0), r_world_body_flu(0, 1), r_world_body_flu(0, 2),
            r_world_body_flu(1, 0), r_world_body_flu(1, 1), r_world_body_flu(1, 2),
            r_world_body_flu(2, 0), r_world_body_flu(2, 1), r_world_body_flu(2, 2));
        tf_rotation.getRPY(roll, pitch, yaw);
        current_yaw = yaw;

        x_real.resize(12);
        x_real << static_cast<float>(position_world.x()),
            static_cast<float>(position_world.y()),
            static_cast<float>(position_world.z()),
            static_cast<float>(velocity_body_flu.x()),
            static_cast<float>(velocity_body_flu.y()),
            static_cast<float>(velocity_body_flu.z()),
            static_cast<float>(roll),
            static_cast<float>(pitch),
            static_cast<float>(yaw),
            static_cast<float>(angular_flu.x()),
            static_cast<float>(angular_flu.y()),
            static_cast<float>(angular_flu.z());
        return true;
    }

    void publishControl()
    {
        if (!has_odom_ || !targetFresh()) {
            return;
        }

        Eigen::VectorXf x_real;
        double current_yaw = 0.0;
        if (!buildState(x_real, current_yaw)) {
            return;
        }

        const auto& target = latest_target_.pose.position;
        Eigen::VectorXf x_des(12);
        x_des << static_cast<float>(target.x),
            static_cast<float>(target.y),
            static_cast<float>(target.z),
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            static_cast<float>(current_yaw),
            0.0f,
            0.0f,
            0.0f;

        Eigen::Vector4f output = controller_.execute(x_des, x_real);

        airsim_ros::RotorPWM pwm;
        pwm.header.stamp = ros::Time::now();
        pwm.rotorPWM0 = clampValue(output[0], min_pwm_, max_pwm_);
        pwm.rotorPWM1 = clampValue(output[1], min_pwm_, max_pwm_);
        pwm.rotorPWM2 = clampValue(output[2], min_pwm_, max_pwm_);
        pwm.rotorPWM3 = clampValue(output[3], min_pwm_, max_pwm_);
        pwm_pub_.publish(pwm);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pwm_pub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber target_sub_;
    ros::ServiceClient takeoff_client_;

    UAVLinearController controller_;
    nav_msgs::Odometry latest_odom_;
    geometry_msgs::PoseStamped latest_target_;
    ros::Time latest_target_time_;
    bool has_odom_ = false;
    bool has_target_ = false;

    std::string odom_topic_;
    std::string target_topic_;
    std::string pwm_topic_;
    std::string takeoff_service_;
    bool auto_takeoff_ = true;
    double control_rate_hz_ = 200.0;
    double max_target_age_s_ = 2.0;
    double post_takeoff_wait_s_ = 3.0;
    double max_pwm_ = 1.0;
    double min_pwm_ = 0.0;
};
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "waypoint_pwm_controller");
    WaypointPwmController controller;
    controller.run();
    return 0;
}
