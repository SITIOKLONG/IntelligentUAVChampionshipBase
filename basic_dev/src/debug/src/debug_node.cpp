#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <airsim_ros/VelCmd.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <cmath>
#include <vector>

namespace
{
constexpr const char* kVizFrame = "world";

geometry_msgs::PoseStamped latest_gps;
bool has_gps = false;
geometry_msgs::PoseStamped latest_pose_ned;
bool has_pose_ned = false;
bool has_pose_gt = false;

ros::Publisher gps_pose_pub;
ros::Publisher gps_path_pub;
ros::Publisher pose_gt_pub;
ros::Publisher pose_gt_path_pub;
ros::Publisher goal_marker_pub;
ros::Publisher lidar_world_pub;
ros::Publisher odom_pose_pub;
ros::Publisher odom_path_pub;
ros::Publisher vel_cmd_marker_pub;
ros::Publisher waypoint_path_pub;
ros::Publisher waypoint_marker_pub;

nav_msgs::Path gps_path;
nav_msgs::Path pose_gt_path;
nav_msgs::Path odom_path;

ros::Time stampOrNow(const ros::Time& stamp)
{
    return stamp.isZero() ? ros::Time::now() : stamp;
}

geometry_msgs::Point nedToWorld(double x, double y, double z)
{
    geometry_msgs::Point p;
    p.x = x;
    p.y = -y;
    p.z = -z;
    return p;
}

tf2::Vector3 nedVectorToWorld(const tf2::Vector3& v)
{
    return tf2::Vector3(v.x(), -v.y(), -v.z());
}

geometry_msgs::Point controlToWorld(double x, double y, double alt_up)
{
    geometry_msgs::Point p;
    p.x = x;
    p.y = -y;
    p.z = alt_up;
    return p;
}

tf2::Quaternion nedToWorldRotation()
{
    tf2::Quaternion q;
    q.setRPY(M_PI, 0.0, 0.0);
    q.normalize();
    return q;
}

geometry_msgs::Quaternion orientationNedToWorld(
    const geometry_msgs::Quaternion& orientation_ned)
{
    tf2::Quaternion q_ned(
        orientation_ned.x,
        orientation_ned.y,
        orientation_ned.z,
        orientation_ned.w);
    if (q_ned.length2() < 1e-12) {
        geometry_msgs::Quaternion identity;
        identity.w = 1.0;
        return identity;
    }

    q_ned.normalize();
    tf2::Quaternion q_world = nedToWorldRotation() * q_ned;
    q_world.normalize();

    geometry_msgs::Quaternion out;
    out.x = q_world.x();
    out.y = q_world.y();
    out.z = q_world.z();
    out.w = q_world.w();
    return out;
}

geometry_msgs::PoseStamped makeWorldPoseFromNed(
    const geometry_msgs::PoseStamped& msg)
{
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = stampOrNow(msg.header.stamp);
    pose.header.frame_id = kVizFrame;
    pose.pose.position = nedToWorld(
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z);
    pose.pose.orientation = orientationNedToWorld(msg.pose.orientation);
    return pose;
}

void updateLatestPoseForDebug(
    const geometry_msgs::PoseStamped& pose_ned,
    bool is_ground_truth)
{
    if (is_ground_truth || !has_pose_gt) {
        latest_pose_ned = pose_ned;
        has_pose_ned = true;
        has_pose_gt = is_ground_truth;
    }
}

geometry_msgs::PoseStamped makeWorldPoseFromOdom(
    const nav_msgs::Odometry& msg)
{
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = stampOrNow(msg.header.stamp);
    pose.header.frame_id = kVizFrame;
    pose.pose.position = nedToWorld(
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z);
    pose.pose.orientation = orientationNedToWorld(msg.pose.pose.orientation);
    return pose;
}

void appendBoundedPath(nav_msgs::Path& path, const geometry_msgs::PoseStamped& pose)
{
    path.header.frame_id = kVizFrame;
    path.header.stamp = pose.header.stamp;
    path.poses.push_back(pose);

    if (path.poses.size() > 2000) {
        path.poses.erase(path.poses.begin());
    }
}

void gpsCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    latest_gps = *msg;
    has_gps = true;
    updateLatestPoseForDebug(*msg, false);

    geometry_msgs::PoseStamped pose = makeWorldPoseFromNed(*msg);
    gps_pose_pub.publish(pose);

    appendBoundedPath(gps_path, pose);
    gps_path_pub.publish(gps_path);
}

void poseGtCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    updateLatestPoseForDebug(*msg, true);

    geometry_msgs::PoseStamped pose = makeWorldPoseFromNed(*msg);
    pose_gt_pub.publish(pose);

    appendBoundedPath(pose_gt_path, pose);
    pose_gt_path_pub.publish(pose_gt_path);
}

void goalCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = kVizFrame;
    marker.header.stamp = stampOrNow(msg->header.stamp);
    marker.ns = "end_goal_world";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position = nedToWorld(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 5.0;
    marker.scale.y = 5.0;
    marker.scale.z = 5.0;
    marker.color.r = 1.0;
    marker.color.a = 1.0;
    goal_marker_pub.publish(marker);
}

void odomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
    geometry_msgs::PoseStamped pose = makeWorldPoseFromOdom(*msg);

    odom_pose_pub.publish(pose);

    appendBoundedPath(odom_path, pose);
    odom_path_pub.publish(odom_path);
}

void velCmdCb(const airsim_ros::VelCmd::ConstPtr& msg)
{
    if (!has_pose_ned) {
        return;
    }

    tf2::Quaternion q_ned(
        latest_pose_ned.pose.orientation.x,
        latest_pose_ned.pose.orientation.y,
        latest_pose_ned.pose.orientation.z,
        latest_pose_ned.pose.orientation.w);
    if (q_ned.length2() < 1e-12) {
        return;
    }
    q_ned.normalize();

    // AirSim vel_body_cmd uses x-forward, y-right, z-up. Convert to body FRD
    // before applying the raw AirSim NED attitude.
    const tf2::Vector3 cmd_body_frd(msg->vx, msg->vy, -msg->vz);
    const tf2::Vector3 cmd_ned = tf2::quatRotate(q_ned, cmd_body_frd);
    const tf2::Vector3 cmd_world = nedVectorToWorld(cmd_ned);

    const double speed = cmd_world.length();
    if (speed < 1e-6) {
        return;
    }

    const geometry_msgs::Point start = nedToWorld(
        latest_pose_ned.pose.position.x,
        latest_pose_ned.pose.position.y,
        latest_pose_ned.pose.position.z);
    const tf2::Vector3 direction = cmd_world.normalized();

    geometry_msgs::Point end = start;
    end.x += 5.0 * direction.x();
    end.y += 5.0 * direction.y();
    end.z += 5.0 * direction.z();

    visualization_msgs::Marker marker;
    marker.header.frame_id = kVizFrame;
    marker.header.stamp = ros::Time::now();
    marker.ns = "vel_cmd_world";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.points.push_back(start);
    marker.points.push_back(end);
    marker.scale.x = 0.25;
    marker.scale.y = 0.7;
    marker.scale.z = 0.9;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.7;
    marker.color.b = 1.0;
    marker.lifetime = ros::Duration(0.25);

    vel_cmd_marker_pub.publish(marker);
}

void waypointsCb(const nav_msgs::Path::ConstPtr& msg)
{
    nav_msgs::Path out;
    out.header.frame_id = kVizFrame;
    out.header.stamp = stampOrNow(msg->header.stamp);

    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker clear_marker;
    clear_marker.header = out.header;
    clear_marker.ns = "waypoints_world";
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    int id = 0;
    for (const auto& pose : msg->poses) {
        geometry_msgs::PoseStamped ps;
        ps.header = out.header;
        ps.pose.position = controlToWorld(
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z);
        ps.pose.orientation.w = 1.0;
        out.poses.push_back(ps);

        visualization_msgs::Marker marker;
        marker.header = out.header;
        marker.ns = "waypoints_world";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position = ps.pose.position;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2.0;
        marker.scale.y = 2.0;
        marker.scale.z = 2.0;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.8;
        marker.color.b = 0.0;
        markers.markers.push_back(marker);
    }

    waypoint_path_pub.publish(out);
    waypoint_marker_pub.publish(markers);
}

void lidarCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    if (!has_gps) {
        return;
    }

    sensor_msgs::PointCloud2 out = *msg;
    out.header.frame_id = kVizFrame;
    out.header.stamp = stampOrNow(msg->header.stamp);

    tf2::Quaternion q(
        latest_gps.pose.orientation.x,
        latest_gps.pose.orientation.y,
        latest_gps.pose.orientation.z,
        latest_gps.pose.orientation.w);
    if (q.length2() < 1e-12) {
        return;
    }
    q.normalize();

    tf2::Transform body_to_world_ned(q);
    tf2::Vector3 pos_ned(
        latest_gps.pose.position.x,
        latest_gps.pose.position.y,
        latest_gps.pose.position.z);

    sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");

    const tf2::Vector3 lidar_offset_body(0.0, 0.0, -0.05);

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        tf2::Vector3 point_lidar(*iter_x, *iter_y, *iter_z);
        tf2::Vector3 point_body = lidar_offset_body + point_lidar;
        tf2::Vector3 point_ned = pos_ned + body_to_world_ned * point_body;
        geometry_msgs::Point point_world = nedToWorld(
            point_ned.x(),
            point_ned.y(),
            point_ned.z());

        *iter_x = static_cast<float>(point_world.x);
        *iter_y = static_cast<float>(point_world.y);
        *iter_z = static_cast<float>(point_world.z);
    }

    lidar_world_pub.publish(out);
}
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "debug_node");
    ros::NodeHandle nh;

    gps_pose_pub =
        nh.advertise<geometry_msgs::PoseStamped>("/debug/gps_pose_world", 10);
    gps_path_pub =
        nh.advertise<nav_msgs::Path>("/debug/gps_path_world", 10);
    pose_gt_pub =
        nh.advertise<geometry_msgs::PoseStamped>("/debug/pose_gt_world", 10);
    pose_gt_path_pub =
        nh.advertise<nav_msgs::Path>("/debug/pose_gt_path_world", 10);
    goal_marker_pub =
        nh.advertise<visualization_msgs::Marker>("/debug/end_goal_world", 1, true);
    lidar_world_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/debug/lidar_world", 1);
    odom_pose_pub =
        nh.advertise<geometry_msgs::PoseStamped>("/debug/odom_world", 10);
    odom_path_pub =
        nh.advertise<nav_msgs::Path>("/debug/odom_path_world", 10);
    vel_cmd_marker_pub =
        nh.advertise<visualization_msgs::Marker>("/debug/vel_cmd_world", 10);
    waypoint_path_pub =
        nh.advertise<nav_msgs::Path>("/debug/waypoints_world", 1, true);
    waypoint_marker_pub =
        nh.advertise<visualization_msgs::MarkerArray>("/debug/waypoint_markers_world", 1, true);

    ros::Subscriber gps_sub =
        nh.subscribe("/airsim_node/drone_1/gps", 10, gpsCb);
    ros::Subscriber pose_gt_sub =
        nh.subscribe("/airsim_node/drone_1/debug/pose_gt", 10, poseGtCb);
    ros::Subscriber goal_sub =
        nh.subscribe("/airsim_node/end_goal", 10, goalCb);
    ros::Subscriber waypoints_sub =
        nh.subscribe("/my_drone/waypoints", 1, waypointsCb);
    ros::Subscriber lidar_sub =
        nh.subscribe("/airsim_node/drone_1/lidar", 1, lidarCb);
    ros::Subscriber odom_sub =
        nh.subscribe("/eskf_odom", 30, odomCb);
    ros::Subscriber vel_cmd_sub =
        nh.subscribe("/airsim_node/drone_1/vel_body_cmd", 30, velCmdCb);

    ROS_INFO("debug_node started; all debug outputs use frame '%s'", kVizFrame);

    ros::spin();
    return 0;
}
