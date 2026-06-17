#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_srvs/Empty.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{
std::vector<geometry_msgs::PoseStamped> waypoints;
std::vector<geometry_msgs::Point> reached_waypoints;
std::size_t current_index = 0;
nav_msgs::Odometry latest_odom;
bool has_odom = false;

ros::Publisher waypoint_path_pub;
ros::Publisher current_waypoint_pub;

double reached_threshold_m = 5.0;
double path_update_match_tolerance_m = 6.0;
double fallback_extension_m = 8.0;
double align_after_reached_s = 1.0;
double search_yaw_deg = 45.0;
double search_sweep_period_s = 3.0;
std::string world_frame_id = "world";
bool aligning = false;
ros::Time align_until;
geometry_msgs::PoseStamped align_target;
bool align_then_search = false;
bool searching_next_door = false;
ros::Time search_started;
geometry_msgs::Point search_hold_position;
tf2::Vector3 search_base_direction(1.0, 0.0, 0.0);

tf2::Vector3 droneForwardWorld();

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

bool alreadyReached(const geometry_msgs::Point& point)
{
    for (const auto& reached : reached_waypoints) {
        if (samePoint(point, reached, reached_threshold_m)) {
            return true;
        }
    }
    return false;
}

geometry_msgs::Point addWorldVector(const geometry_msgs::Point& point, const tf2::Vector3& direction, double distance)
{
    geometry_msgs::Point out = point;
    out.x += direction.x() * distance;
    out.y += direction.y() * distance;
    out.z += direction.z() * distance;
    return out;
}

tf2::Vector3 normalizedDirection(const geometry_msgs::Point& from, const geometry_msgs::Point& to)
{
    tf2::Vector3 direction(to.x - from.x, to.y - from.y, to.z - from.z);
    if (direction.length2() < 1e-6) {
        return tf2::Vector3(1.0, 0.0, 0.0);
    }
    return direction.normalized();
}

tf2::Vector3 directionWithYawOffset(const tf2::Vector3& base_direction, double yaw_offset)
{
    tf2::Vector3 base = base_direction;
    if (base.length2() < 1e-6) {
        base = tf2::Vector3(1.0, 0.0, 0.0);
    }
    base.normalize();

    double horizontal_norm = std::hypot(base.x(), base.y());
    if (horizontal_norm < 1e-6) {
        horizontal_norm = 1.0;
    }
    const double yaw = std::atan2(base.y(), base.x()) + yaw_offset;
    tf2::Vector3 direction(horizontal_norm * std::cos(yaw), horizontal_norm * std::sin(yaw), base.z());
    if (direction.length2() < 1e-6) {
        return tf2::Vector3(1.0, 0.0, 0.0);
    }
    return direction.normalized();
}

geometry_msgs::Quaternion orientationFacingDirection(const tf2::Vector3& direction)
{
    tf2::Vector3 x_axis = direction;
    if (x_axis.length2() < 1e-6) {
        x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    }
    x_axis.normalize();

    tf2::Vector3 up(0.0, 0.0, 1.0);
    if (std::abs(x_axis.dot(up)) > 0.98) {
        up = tf2::Vector3(0.0, 1.0, 0.0);
    }
    tf2::Vector3 y_axis = up.cross(x_axis).normalized();
    tf2::Vector3 z_axis = x_axis.cross(y_axis).normalized();

    tf2::Matrix3x3 rotation(
        x_axis.x(), y_axis.x(), z_axis.x(),
        x_axis.y(), y_axis.y(), z_axis.y(),
        x_axis.z(), y_axis.z(), z_axis.z());
    tf2::Quaternion q;
    rotation.getRotation(q);
    q.normalize();

    geometry_msgs::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
}

bool hasValidOrientation(const geometry_msgs::Quaternion& q)
{
    const double norm2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    return norm2 > 1e-12;
}

void updateWaypointOrientations()
{
    if (waypoints.empty()) {
        return;
    }
    for (std::size_t i = 0; i < waypoints.size(); ++i) {
        if (hasValidOrientation(waypoints[i].pose.orientation)) {
            continue;
        }
        tf2::Vector3 direction = droneForwardWorld();
        if (i + 1 < waypoints.size()) {
            direction = normalizedDirection(waypoints[i].pose.position, waypoints[i + 1].pose.position);
        } else if (i > 0) {
            direction = normalizedDirection(waypoints[i - 1].pose.position, waypoints[i].pose.position);
        }
        waypoints[i].pose.orientation = orientationFacingDirection(direction);
    }
}

tf2::Vector3 droneForwardWorld()
{
    const auto& q_msg = latest_odom.pose.pose.orientation;
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    if (q.length2() < 1e-12) {
        return tf2::Vector3(1.0, 0.0, 0.0);
    }
    q.normalize();
    const tf2::Vector3 forward_ned = tf2::quatRotate(q, tf2::Vector3(1.0, 0.0, 0.0));
    tf2::Vector3 forward_world(forward_ned.x(), -forward_ned.y(), -forward_ned.z());
    forward_world.setZ(0.0);
    if (forward_world.length2() < 1e-6) {
        return tf2::Vector3(1.0, 0.0, 0.0);
    }
    return forward_world.normalized();
}

void appendFallbackWaypoint()
{
    if (waypoints.empty()) {
        return;
    }
    tf2::Vector3 direction = droneForwardWorld();
    if (waypoints.size() >= 2) {
        direction = normalizedDirection(
            waypoints[waypoints.size() - 2].pose.position,
            waypoints.back().pose.position);
    }

    geometry_msgs::PoseStamped fallback = waypoints.back();
    fallback.header.stamp = ros::Time::now();
    fallback.header.frame_id = world_frame_id;
    fallback.pose.position = addWorldVector(waypoints.back().pose.position, direction, fallback_extension_m);
    fallback.pose.orientation = orientationFacingDirection(direction);
    waypoints.push_back(fallback);
    updateWaypointOrientations();
}

void publishPath()
{
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = world_frame_id;
    path.poses = waypoints;
    for (auto& pose : path.poses) {
        pose.header = path.header;
        if (!hasValidOrientation(pose.pose.orientation)) {
            pose.pose.orientation.w = 1.0;
        }
    }
    waypoint_path_pub.publish(path);
}

void publishCurrentWaypoint()
{
    if (searching_next_door) {
        geometry_msgs::PoseStamped target;
        target.header.stamp = ros::Time::now();
        target.header.frame_id = world_frame_id;
        target.pose.position = search_hold_position;

        const double period = std::max(0.1, search_sweep_period_s);
        const double elapsed = (ros::Time::now() - search_started).toSec();
        const double yaw_offset =
            (search_yaw_deg * M_PI / 180.0) * std::sin(2.0 * M_PI * elapsed / period);
        target.pose.orientation = orientationFacingDirection(
            directionWithYawOffset(search_base_direction, yaw_offset));
        current_waypoint_pub.publish(target);
        return;
    }

    if (aligning) {
        geometry_msgs::PoseStamped target = align_target;
        target.header.stamp = ros::Time::now();
        target.header.frame_id = world_frame_id;
        current_waypoint_pub.publish(target);
        return;
    }

    if (waypoints.empty()) {
        return;
    }

    current_index = std::min(current_index, waypoints.size() - 1);
    geometry_msgs::PoseStamped target = waypoints[current_index];
    target.header.stamp = ros::Time::now();
    target.header.frame_id = world_frame_id;
    if (!hasValidOrientation(target.pose.orientation)) {
        target.pose.orientation.w = 1.0;
    }
    current_waypoint_pub.publish(target);
}

void beginAlignBeforeNext(const geometry_msgs::Point& hold_position, const tf2::Vector3& direction)
{
    aligning = align_after_reached_s > 0.0;
    if (!aligning) {
        return;
    }

    align_until = ros::Time::now() + ros::Duration(align_after_reached_s);
    align_target.header.stamp = ros::Time::now();
    align_target.header.frame_id = world_frame_id;
    align_target.pose.position = hold_position;
    align_target.pose.orientation = orientationFacingDirection(direction);
}

void beginSearchForNextDoor(const geometry_msgs::Point& hold_position, const tf2::Vector3& base_direction)
{
    searching_next_door = true;
    search_started = ros::Time::now();
    search_hold_position = hold_position;
    search_base_direction = base_direction.length2() < 1e-6 ? droneForwardWorld() : base_direction.normalized();
}

void beginAlignThenSearchForNextDoor(const geometry_msgs::Point& hold_position, const tf2::Vector3& base_direction)
{
    search_hold_position = hold_position;
    search_base_direction = base_direction.length2() < 1e-6 ? droneForwardWorld() : base_direction.normalized();
    searching_next_door = false;
    align_then_search = align_after_reached_s > 0.0;

    if (align_then_search) {
        beginAlignBeforeNext(hold_position, search_base_direction);
    } else {
        beginSearchForNextDoor(hold_position, search_base_direction);
    }
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

    std::vector<geometry_msgs::PoseStamped> new_waypoints = msg->poses;
    for (auto& pose : new_waypoints) {
        pose.header.frame_id = world_frame_id;
    }
    new_waypoints.erase(
        std::remove_if(
            new_waypoints.begin(),
            new_waypoints.end(),
            [](const geometry_msgs::PoseStamped& pose) {
                return alreadyReached(pose.pose.position);
            }),
        new_waypoints.end());
    if (new_waypoints.empty()) {
        return;
    }

    if (!had_target) {
        waypoints = new_waypoints;
        current_index = 0;
    } else if (alreadyReached(old_target)) {
        waypoints = new_waypoints;
        current_index = 0;
    } else {
        std::size_t matched_index = 0;
        double best_distance = distance3d(old_target, new_waypoints.front().pose.position);
        for (std::size_t i = 1; i < new_waypoints.size(); ++i) {
            const double d = distance3d(old_target, new_waypoints[i].pose.position);
            if (d < best_distance) {
                best_distance = d;
                matched_index = i;
            }
        }
        if (best_distance > path_update_match_tolerance_m) {
            ROS_WARN_THROTTLE(
                1.0,
                "my_drone: ignoring path update; current target moved %.1fm away",
                best_distance);
            return;
        }
        waypoints = new_waypoints;
        current_index = matched_index;
    }

    updateWaypointOrientations();
    align_then_search = false;
    searching_next_door = false;
    publishPath();
    publishCurrentWaypoint();
}

void addWaypointCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped wp = *msg;
    wp.header.frame_id = world_frame_id;
    wp.header.stamp = ros::Time::now();
    if (!hasValidOrientation(wp.pose.orientation)) {
        wp.pose.orientation.w = 1.0;
    }
    waypoints.push_back(wp);
    updateWaypointOrientations();
    publishPath();
    publishCurrentWaypoint();
}

bool clearWaypointsCb(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
    waypoints.clear();
    reached_waypoints.clear();
    current_index = 0;
    aligning = false;
    align_then_search = false;
    searching_next_door = false;
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

    if (searching_next_door) {
        publishCurrentWaypoint();
        return;
    }

    if (aligning) {
        if (ros::Time::now() >= align_until) {
            aligning = false;
            if (align_then_search) {
                align_then_search = false;
                beginSearchForNextDoor(search_hold_position, search_base_direction);
                publishCurrentWaypoint();
                return;
            }
        } else {
            publishCurrentWaypoint();
            return;
        }
    }

    const double distance = distance3d(current_world, waypoints[current_index].pose.position);

    if (distance < reached_threshold_m && current_index + 1 < waypoints.size()) {
        const tf2::Vector3 next_direction =
            normalizedDirection(waypoints[current_index].pose.position, waypoints[current_index + 1].pose.position);
        if (!alreadyReached(waypoints[current_index].pose.position)) {
            reached_waypoints.push_back(waypoints[current_index].pose.position);
        }
        beginAlignBeforeNext(current_world, next_direction);
        ++current_index;
        ROS_INFO(
            "my_drone: reached waypoint, aligning %.2fs before switching to #%zu/%zu",
            align_after_reached_s,
            current_index + 1,
            waypoints.size());
    } else if (distance < reached_threshold_m && current_index + 1 >= waypoints.size()) {
        const geometry_msgs::Point reached_position = waypoints[current_index].pose.position;
        if (!alreadyReached(waypoints[current_index].pose.position)) {
            reached_waypoints.push_back(waypoints[current_index].pose.position);
        }
        tf2::Vector3 search_direction = droneForwardWorld();
        if (current_index > 0) {
            search_direction = normalizedDirection(waypoints[current_index - 1].pose.position, reached_position);
        }
        beginAlignThenSearchForNextDoor(reached_position, search_direction);
        ROS_WARN_THROTTLE(
            1.0,
            "my_drone: no next door yet; aligning then searching +/-%.1f deg",
            search_yaw_deg);
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
    pnh.param("path_update_match_tolerance_m", path_update_match_tolerance_m, 6.0);
    pnh.param("fallback_extension_m", fallback_extension_m, 8.0);
    pnh.param("align_after_reached_s", align_after_reached_s, 1.0);
    pnh.param("search_yaw_deg", search_yaw_deg, 45.0);
    pnh.param("search_sweep_period_s", search_sweep_period_s, 3.0);
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
