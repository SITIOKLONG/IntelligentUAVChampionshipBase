#include <ros/ros.h>

#include <Eigen/Dense>
#include <airsim_ros/Takeoff.h>
#include <airsim_ros/VelCmd.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace
{
double clampValue(double value, double min_value, double max_value)
{
    return std::max(min_value, std::min(max_value, value));
}

Eigen::Vector3d worldToNedVector(const Eigen::Vector3d& world)
{
    return Eigen::Vector3d(world.x(), -world.y(), -world.z());
}

Eigen::Vector3d nedToWorldPoint(const Eigen::Vector3d& ned)
{
    return Eigen::Vector3d(ned.x(), -ned.y(), -ned.z());
}

Eigen::Vector3d nedToWorldVector(const Eigen::Vector3d& ned)
{
    return Eigen::Vector3d(ned.x(), -ned.y(), -ned.z());
}

double wrapAngle(double angle)
{
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

bool validQuaternion(const geometry_msgs::Quaternion& q)
{
    const double norm2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    return norm2 > 1e-12;
}

bool yawFromTargetOrientation(const geometry_msgs::Quaternion& q_msg, double& yaw)
{
    if (!validQuaternion(q_msg)) {
        return false;
    }
    Eigen::Quaterniond q_world_body(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    q_world_body.normalize();
    const Eigen::Vector3d forward_world = q_world_body.toRotationMatrix() * Eigen::Vector3d::UnitX();
    const double horizontal_norm = std::hypot(forward_world.x(), forward_world.y());
    if (horizontal_norm < 1e-6) {
        return false;
    }
    yaw = std::atan2(forward_world.y(), forward_world.x());
    return true;
}

class WaypointVelController
{
public:
    WaypointVelController()
        : nh_()
        , pnh_("~")
    {
        pnh_.param<std::string>("odom_topic", odom_topic_, "/eskf_odom");
        pnh_.param<std::string>("target_topic", target_topic_, "/my_drone/current_waypoint");
        pnh_.param<std::string>("cmd_topic", cmd_topic_, "/airsim_node/drone_1/vel_body_cmd");
        pnh_.param<std::string>("takeoff_service", takeoff_service_, "/airsim_node/drone_1/takeoff");
        pnh_.param("auto_takeoff", auto_takeoff_, true);
        pnh_.param("startup_forward_s", startup_forward_s_, 3.0);
        pnh_.param("startup_forward_vx", startup_forward_vx_, 2.0);
        pnh_.param("control_rate_hz", control_rate_hz_, 30.0);
        pnh_.param("max_target_age_s", max_target_age_s_, 2.0);
        pnh_.param("kp_xy", kp_xy_, 0.45);
        pnh_.param("kp_z", kp_z_, 0.6);
        pnh_.param("kp_yaw", kp_yaw_, 45.0);
        pnh_.param("max_xy_speed", max_xy_speed_, 4.0);
        pnh_.param("max_z_speed", max_z_speed_, 2.0);
        pnh_.param("max_yaw_rate", max_yaw_rate_, 30.0);

        cmd_pub_ = nh_.advertise<airsim_ros::VelCmd>(cmd_topic_, 10);
        odom_sub_ = nh_.subscribe(odom_topic_, 1, &WaypointVelController::odomCallback, this);
        target_sub_ = nh_.subscribe(target_topic_, 1, &WaypointVelController::targetCallback, this);

        if (auto_takeoff_) {
            takeoff_client_ = nh_.serviceClient<airsim_ros::Takeoff>(takeoff_service_);
            ROS_INFO("waypoint_vel_controller: waiting for takeoff service %s", takeoff_service_.c_str());
            takeoff_client_.waitForExistence();
            airsim_ros::Takeoff srv;
            srv.request.waitOnLastTask = 0;
            if (takeoff_client_.call(srv)) {
                ROS_INFO("waypoint_vel_controller: takeoff requested");
                startup_forward_until_ = ros::Time::now() + ros::Duration(startup_forward_s_);
            } else {
                ROS_WARN("waypoint_vel_controller: takeoff service call failed");
            }
        }

        ROS_INFO(
            "waypoint_vel_controller: odom=%s target=%s cmd=%s kp=(%.2f, %.2f) max=(%.2f, %.2f)",
            odom_topic_.c_str(),
            target_topic_.c_str(),
            cmd_topic_.c_str(),
            kp_xy_,
            kp_z_,
            max_xy_speed_,
            max_z_speed_);
    }

    void run()
    {
        ros::Rate rate(control_rate_hz_);
        while (ros::ok()) {
            ros::spinOnce();
            publishCommand();
            rate.sleep();
        }
        publishHover(true);
    }

private:
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        latest_odom_ = *msg;
        has_odom_ = true;
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        latest_target_ = *msg;
        latest_target_time_ = ros::Time::now();
        has_target_ = true;
    }

    bool targetFresh() const
    {
        return has_target_ &&
               (max_target_age_s_ <= 0.0 ||
                (ros::Time::now() - latest_target_time_).toSec() <= max_target_age_s_);
    }

    void publishHover(bool hard_stop)
    {
        airsim_ros::VelCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "body";
        cmd.vx = 0.0;
        cmd.vy = 0.0;
        cmd.vz = 0.0;
        cmd.yawRate = 0.0;
        cmd.va = 4;
        cmd.stop = hard_stop ? 1 : 0;
        cmd_pub_.publish(cmd);
    }

    void publishCommand()
    {
        if (startup_forward_until_ > ros::Time::now()) {
            publishStartupForward();
            return;
        }
        if (!has_odom_ || !targetFresh()) {
            publishHover(false);
            return;
        }

        const auto& pose = latest_odom_.pose.pose;
        Eigen::Quaterniond q_ned_body(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);
        if (q_ned_body.norm() < 1e-12) {
            publishHover(false);
            return;
        }
        q_ned_body.normalize();

        const Eigen::Vector3d current_ned(
            pose.position.x,
            pose.position.y,
            pose.position.z);
        const Eigen::Vector3d current_world = nedToWorldPoint(current_ned);
        const Eigen::Vector3d target_world(
            latest_target_.pose.position.x,
            latest_target_.pose.position.y,
            latest_target_.pose.position.z);

        const Eigen::Vector3d error_world = target_world - current_world;
        const Eigen::Vector3d desired_vel_world(
            clampValue(kp_xy_ * error_world.x(), -max_xy_speed_, max_xy_speed_),
            clampValue(kp_xy_ * error_world.y(), -max_xy_speed_, max_xy_speed_),
            clampValue(kp_z_ * error_world.z(), -max_z_speed_, max_z_speed_));

        const Eigen::Vector3d desired_vel_ned = worldToNedVector(desired_vel_world);
        const Eigen::Vector3d desired_vel_body_frd =
            q_ned_body.toRotationMatrix().transpose() * desired_vel_ned;
        const Eigen::Vector3d forward_world =
            nedToWorldVector(q_ned_body.toRotationMatrix() * Eigen::Vector3d::UnitX());
        const double current_yaw = std::atan2(forward_world.y(), forward_world.x());
        const double horizontal_error = std::hypot(error_world.x(), error_world.y());
        double target_yaw = 0.0;
        bool has_target_yaw = yawFromTargetOrientation(latest_target_.pose.orientation, target_yaw);
        if (!has_target_yaw && horizontal_error > 0.5) {
            target_yaw = std::atan2(error_world.y(), error_world.x());
            has_target_yaw = true;
        }
        const double yaw_error = has_target_yaw ? wrapAngle(target_yaw - current_yaw) : 0.0;

        airsim_ros::VelCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "body";
        cmd.vx = desired_vel_body_frd.x();
        cmd.vy = desired_vel_body_frd.y();
        cmd.vz = -desired_vel_body_frd.z();
        cmd.yawRate = -clampValue(kp_yaw_ * yaw_error, -max_yaw_rate_, max_yaw_rate_);
        cmd.va = 5;
        cmd.stop = 0;
        cmd_pub_.publish(cmd);
    }

    void publishStartupForward()
    {
        airsim_ros::VelCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "body";
        cmd.vx = startup_forward_vx_;
        cmd.vy = 0.0;
        cmd.vz = 0.0;
        cmd.yawRate = 0.0;
        cmd.va = 5;
        cmd.stop = 0;
        cmd_pub_.publish(cmd);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber target_sub_;
    ros::ServiceClient takeoff_client_;

    nav_msgs::Odometry latest_odom_;
    geometry_msgs::PoseStamped latest_target_;
    ros::Time latest_target_time_;
    bool has_odom_ = false;
    bool has_target_ = false;

    std::string odom_topic_;
    std::string target_topic_;
    std::string cmd_topic_;
    std::string takeoff_service_;
    bool auto_takeoff_ = true;
    ros::Time startup_forward_until_;
    double startup_forward_s_ = 3.0;
    double startup_forward_vx_ = 2.0;
    double control_rate_hz_ = 30.0;
    double max_target_age_s_ = 2.0;
    double kp_xy_ = 0.45;
    double kp_z_ = 0.6;
    double kp_yaw_ = 45.0;
    double max_xy_speed_ = 4.0;
    double max_z_speed_ = 2.0;
    double max_yaw_rate_ = 30.0;
};
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "waypoint_vel_controller");
    WaypointVelController controller;
    controller.run();
    return 0;
}
