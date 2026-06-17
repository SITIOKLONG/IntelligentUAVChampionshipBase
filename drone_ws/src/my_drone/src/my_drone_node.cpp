#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_srvs/Empty.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
std::vector<geometry_msgs::PoseStamped> waypoints;
std::size_t current_index = 0;
nav_msgs::Odometry latest_odom;
bool has_odom = false;

ros::Publisher waypoint_path_pub;
ros::Publisher current_waypoint_pub;

double reached_threshold_m = 5.0;
std::string world_frame_id = "world";

geometry_msgs::Point nedToWorld(const geometry_msgs::Point& ned)
{
    geometry_msgs::Point world;
    world.x = ned.x;
    world.y = -ned.y;
    world.z = -ned.z;
    return world;
}

double distance3d(const geometry_msgs::Point& a, const geometry_msgs::Point& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool samePoint(const geometry_msgs::Point& a, const geometry_msgs::Point& b, double tolerance)
{
    return distance3d(a, b) <= tolerance;
}

void publishPath()
{
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = world_frame_id;
    path.poses = waypoints;
    for (auto& pose : path.poses) {
        pose.header = path.header;
        if (std::abs(pose.pose.orientation.w) < 1e-12 &&
            std::abs(pose.pose.orientation.x) < 1e-12 &&
            std::abs(pose.pose.orientation.y) < 1e-12 &&
            std::abs(pose.pose.orientation.z) < 1e-12) {
            pose.pose.orientation.w = 1.0;
        }
    }
    waypoint_path_pub.publish(path);
}

void publishCurrentWaypoint()
{
    if (waypoints.empty()) {
        return;
    }

    current_index = std::min(current_index, waypoints.size() - 1);
    geometry_msgs::PoseStamped target = waypoints[current_index];
    target.header.stamp = ros::Time::now();
    target.header.frame_id = world_frame_id;
    if (std::abs(target.pose.orientation.w) < 1e-12 &&
        std::abs(target.pose.orientation.x) < 1e-12 &&
        std::abs(target.pose.orientation.y) < 1e-12 &&
        std::abs(target.pose.orientation.z) < 1e-12) {
        target.pose.orientation.w = 1.0;
    }
    current_waypoint_pub.publish(target);
}

void odomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
    latest_odom = *msg;
    has_odom = true;
}

void pathCb(const nav_msgs::Path::ConstPtr& msg)
{
    if (msg->poses.empty()) {
        return;
    }

    geometry_msgs::Point old_target;
    const bool had_target = !waypoints.empty() && current_index < waypoints.size();
    if (had_target) {
        old_target = waypoints[current_index].pose.position;
    }

    waypoints = msg->poses;
    for (auto& pose : waypoints) {
        pose.header.frame_id = world_frame_id;
    }

    if (!had_target) {
        current_index = 0;
    } else if (current_index >= waypoints.size() ||
               !samePoint(old_target, waypoints[current_index].pose.position, 2.0)) {
        current_index = 0;
        double best_distance = distance3d(old_target, waypoints.front().pose.position);
        for (std::size_t i = 1; i < waypoints.size(); ++i) {
            const double d = distance3d(old_target, waypoints[i].pose.position);
            if (d < best_distance) {
                best_distance = d;
                current_index = i;
            }
        }
    }

    publishPath();
    publishCurrentWaypoint();
}

void addWaypointCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped wp = *msg;
    wp.header.frame_id = world_frame_id;
    wp.header.stamp = ros::Time::now();
    if (std::abs(wp.pose.orientation.w) < 1e-12 &&
        std::abs(wp.pose.orientation.x) < 1e-12 &&
        std::abs(wp.pose.orientation.y) < 1e-12 &&
        std::abs(wp.pose.orientation.z) < 1e-12) {
        wp.pose.orientation.w = 1.0;
    }
    waypoints.push_back(wp);
    publishPath();
    publishCurrentWaypoint();
}

bool clearWaypointsCb(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
    waypoints.clear();
    current_index = 0;
    publishPath();
    return true;
}

void timerCb(const ros::TimerEvent&)
{
    if (!has_odom || waypoints.empty()) {
        return;
    }

    current_index = std::min(current_index, waypoints.size() - 1);
    const geometry_msgs::Point current_world = nedToWorld(latest_odom.pose.pose.position);
    const double distance = distance3d(current_world, waypoints[current_index].pose.position);

    if (distance < reached_threshold_m && current_index + 1 < waypoints.size()) {
        ++current_index;
        ROS_INFO(
            "my_drone: reached waypoint, switching to #%zu/%zu",
            current_index + 1,
            waypoints.size());
    }

    publishCurrentWaypoint();
}
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "my_drone_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string input_path_topic;
    std::string odom_topic;
    pnh.param<std::string>("input_path_topic", input_path_topic, "/door_waypoint_detector/waypoints_world");
    pnh.param<std::string>("odom_topic", odom_topic, "/eskf_odom");
    pnh.param("reached_threshold_m", reached_threshold_m, 5.0);
    pnh.param<std::string>("world_frame_id", world_frame_id, "world");

    waypoint_path_pub = nh.advertise<nav_msgs::Path>("/my_drone/waypoints", 1, true);
    current_waypoint_pub = nh.advertise<geometry_msgs::PoseStamped>("/my_drone/current_waypoint", 1, true);

    ros::Subscriber path_sub = nh.subscribe(input_path_topic, 1, pathCb);
    ros::Subscriber odom_sub = nh.subscribe(odom_topic, 1, odomCb);
    ros::Subscriber add_waypoint_sub = nh.subscribe("/my_drone/add_waypoint", 10, addWaypointCb);
    ros::ServiceServer clear_waypoints_srv = nh.advertiseService("/my_drone/clear_waypoints", clearWaypointsCb);
    ros::Timer timer = nh.createTimer(ros::Duration(0.05), timerCb);

    ROS_INFO(
        "my_drone: path=%s odom=%s threshold=%.2fm",
        input_path_topic.c_str(),
        odom_topic.c_str(),
        reached_threshold_m);

    ros::spin();
    return 0;
}
