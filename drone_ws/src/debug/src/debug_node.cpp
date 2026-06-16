#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <airsim_ros/VelCmd.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
constexpr const char* kVizFrame = "world";

geometry_msgs::PoseStamped latest_gps;
bool has_gps = false;
nav_msgs::Odometry latest_odom;
bool has_odom = false;
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
ros::Publisher front_left_image_plane_pub;
ros::Publisher front_right_image_plane_pub;
ros::Publisher back_left_image_plane_pub;
ros::Publisher back_right_image_plane_pub;

nav_msgs::Path gps_path;
nav_msgs::Path pose_gt_path;
nav_msgs::Path odom_path;
ros::Time last_front_left_image_plane;
ros::Time last_front_right_image_plane;
ros::Time last_back_left_image_plane;
ros::Time last_back_right_image_plane;

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

geometry_msgs::Point pointBodyFrdToWorld(
    const tf2::Vector3& point_body_frd,
    const geometry_msgs::PoseStamped& pose_ned)
{
    tf2::Quaternion q_ned(
        pose_ned.pose.orientation.x,
        pose_ned.pose.orientation.y,
        pose_ned.pose.orientation.z,
        pose_ned.pose.orientation.w);
    if (q_ned.length2() < 1e-12) {
        return nedToWorld(
            pose_ned.pose.position.x,
            pose_ned.pose.position.y,
            pose_ned.pose.position.z);
    }
    q_ned.normalize();

    const tf2::Vector3 pos_ned(
        pose_ned.pose.position.x,
        pose_ned.pose.position.y,
        pose_ned.pose.position.z);
    const tf2::Vector3 point_ned = pos_ned + tf2::quatRotate(q_ned, point_body_frd);
    return nedToWorld(point_ned.x(), point_ned.y(), point_ned.z());
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

visualization_msgs::Marker makeBaseMarker(
    const std::string& ns,
    int id,
    int type,
    const ros::Time& stamp)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = kVizFrame;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.lifetime = ros::Duration(0.25);
    return marker;
}

void addSensorSphere(
    visualization_msgs::MarkerArray& markers,
    int& id,
    const std::string& ns,
    const tf2::Vector3& point_body_frd,
    double size,
    double r,
    double g,
    double b,
    const ros::Time& stamp)
{
    visualization_msgs::Marker marker = makeBaseMarker(ns, id++, visualization_msgs::Marker::SPHERE, stamp);
    marker.pose.position = pointBodyFrdToWorld(point_body_frd, latest_pose_ned);
    marker.scale.x = size;
    marker.scale.y = size;
    marker.scale.z = size;
    marker.color.a = 1.0;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    markers.markers.push_back(marker);
}

void addCameraFrustum(
    visualization_msgs::MarkerArray& markers,
    int& id,
    const std::string& ns,
    const tf2::Vector3& camera_center_body_frd,
    bool forward,
    double r,
    double g,
    double b,
    const ros::Time& stamp)
{
    const double length = 1.0;
    const double half = length * std::tan(M_PI / 6.0);
    const double dir = forward ? 1.0 : -1.0;

    const tf2::Vector3 origin = camera_center_body_frd;
    const tf2::Vector3 c1 = origin + tf2::Vector3(dir * length, half, half);
    const tf2::Vector3 c2 = origin + tf2::Vector3(dir * length, -half, half);
    const tf2::Vector3 c3 = origin + tf2::Vector3(dir * length, -half, -half);
    const tf2::Vector3 c4 = origin + tf2::Vector3(dir * length, half, -half);

    visualization_msgs::Marker marker = makeBaseMarker(ns, id++, visualization_msgs::Marker::LINE_LIST, stamp);
    marker.scale.x = 0.03;
    marker.color.a = 1.0;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;

    const auto addLine = [&](const tf2::Vector3& a, const tf2::Vector3& b_point) {
        marker.points.push_back(pointBodyFrdToWorld(a, latest_pose_ned));
        marker.points.push_back(pointBodyFrdToWorld(b_point, latest_pose_ned));
    };

    addLine(origin, c1);
    addLine(origin, c2);
    addLine(origin, c3);
    addLine(origin, c4);
    addLine(c1, c2);
    addLine(c2, c3);
    addLine(c3, c4);
    addLine(c4, c1);

    markers.markers.push_back(marker);
}

void addCameraObject(
    visualization_msgs::MarkerArray& markers,
    int& id,
    const std::string& name,
    const tf2::Vector3& camera_body_frd,
    bool forward,
    double r,
    double g,
    double b,
    const ros::Time& stamp)
{
    const double dir = forward ? 1.0 : -1.0;
    const tf2::Vector3 optical_tip = camera_body_frd + tf2::Vector3(dir * 0.22, 0.0, 0.0);

    visualization_msgs::Marker body =
        makeBaseMarker("sensor_rig_camera_" + name, id++, visualization_msgs::Marker::CUBE, stamp);
    body.pose.position = pointBodyFrdToWorld(camera_body_frd, latest_pose_ned);
    body.scale.x = 0.12;
    body.scale.y = 0.16;
    body.scale.z = 0.10;
    body.color.a = 1.0;
    body.color.r = r;
    body.color.g = g;
    body.color.b = b;
    markers.markers.push_back(body);

    visualization_msgs::Marker lens =
        makeBaseMarker("sensor_rig_camera_lens_" + name, id++, visualization_msgs::Marker::SPHERE, stamp);
    lens.pose.position = pointBodyFrdToWorld(
        camera_body_frd + tf2::Vector3(dir * 0.07, 0.0, 0.0),
        latest_pose_ned);
    lens.scale.x = 0.06;
    lens.scale.y = 0.09;
    lens.scale.z = 0.09;
    lens.color.a = 1.0;
    lens.color.r = 0.02;
    lens.color.g = 0.02;
    lens.color.b = 0.02;
    markers.markers.push_back(lens);

    visualization_msgs::Marker direction =
        makeBaseMarker("sensor_rig_camera_dir_" + name, id++, visualization_msgs::Marker::ARROW, stamp);
    direction.points.push_back(pointBodyFrdToWorld(camera_body_frd, latest_pose_ned));
    direction.points.push_back(pointBodyFrdToWorld(optical_tip, latest_pose_ned));
    direction.scale.x = 0.025;
    direction.scale.y = 0.06;
    direction.scale.z = 0.08;
    direction.color.a = 1.0;
    direction.color.r = r;
    direction.color.g = g;
    direction.color.b = b;
    markers.markers.push_back(direction);

    addCameraFrustum(
        markers,
        id,
        "sensor_rig_camera_fov_" + name,
        camera_body_frd,
        forward,
        r,
        g,
        b,
        stamp);
}

geometry_msgs::Quaternion toMsg(const tf2::Quaternion& q)
{
    geometry_msgs::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
}

void addTransform(
    std::vector<geometry_msgs::TransformStamped>& transforms,
    const ros::Time& stamp,
    const std::string& parent,
    const std::string& child,
    const tf2::Vector3& translation,
    const tf2::Quaternion& rotation)
{
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = parent;
    transform.child_frame_id = child;
    transform.transform.translation.x = translation.x();
    transform.transform.translation.y = translation.y();
    transform.transform.translation.z = translation.z();
    transform.transform.rotation = toMsg(rotation);
    transforms.push_back(transform);
}

tf2::Quaternion cameraLinkToOpticalRotation()
{
    tf2::Matrix3x3 rotation;
    rotation[0] = tf2::Vector3(0.0, 0.0, 1.0);
    rotation[1] = tf2::Vector3(1.0, 0.0, 0.0);
    rotation[2] = tf2::Vector3(0.0, 1.0, 0.0);

    tf2::Quaternion q;
    rotation.getRotation(q);
    q.normalize();
    return q;
}

void publishSensorRigTf()
{
    if (!has_pose_ned) {
        return;
    }

    static tf2_ros::TransformBroadcaster broadcaster;
    const ros::Time stamp = ros::Time::now();
    std::vector<geometry_msgs::TransformStamped> transforms;

    geometry_msgs::Point body_world = nedToWorld(
        latest_pose_ned.pose.position.x,
        latest_pose_ned.pose.position.y,
        latest_pose_ned.pose.position.z);

    tf2::Quaternion body_rotation(
        latest_pose_ned.pose.orientation.x,
        latest_pose_ned.pose.orientation.y,
        latest_pose_ned.pose.orientation.z,
        latest_pose_ned.pose.orientation.w);
    if (body_rotation.length2() < 1e-12) {
        body_rotation.setRPY(0.0, 0.0, 0.0);
    }
    body_rotation.normalize();
    body_rotation = nedToWorldRotation() * body_rotation;
    body_rotation.normalize();

    addTransform(
        transforms,
        stamp,
        kVizFrame,
        "debug/body_frd",
        tf2::Vector3(body_world.x, body_world.y, body_world.z),
        body_rotation);

    tf2::Quaternion identity;
    identity.setRPY(0.0, 0.0, 0.0);
    tf2::Quaternion rear_rotation;
    rear_rotation.setRPY(0.0, 0.0, M_PI);
    rear_rotation.normalize();
    const tf2::Quaternion optical_rotation = cameraLinkToOpticalRotation();

    addTransform(transforms, stamp, "debug/body_frd", "debug/front_left_camera_link", tf2::Vector3(0.175, -0.15, 0.0), identity);
    addTransform(transforms, stamp, "debug/body_frd", "debug/front_right_camera_link", tf2::Vector3(0.175, 0.15, 0.0), identity);
    addTransform(transforms, stamp, "debug/body_frd", "debug/back_left_camera_link", tf2::Vector3(-0.175, -0.15, 0.0), rear_rotation);
    addTransform(transforms, stamp, "debug/body_frd", "debug/back_right_camera_link", tf2::Vector3(-0.175, 0.15, 0.0), rear_rotation);
    addTransform(transforms, stamp, "debug/body_frd", "debug/mid360_lidar_link", tf2::Vector3(0.0, 0.0, -0.05), identity);

    addTransform(transforms, stamp, "debug/front_left_camera_link", "debug/front_left_camera_optical", tf2::Vector3(0.0, 0.0, 0.0), optical_rotation);
    addTransform(transforms, stamp, "debug/front_right_camera_link", "debug/front_right_camera_optical", tf2::Vector3(0.0, 0.0, 0.0), optical_rotation);
    addTransform(transforms, stamp, "debug/back_left_camera_link", "debug/back_left_camera_optical", tf2::Vector3(0.0, 0.0, 0.0), optical_rotation);
    addTransform(transforms, stamp, "debug/back_right_camera_link", "debug/back_right_camera_optical", tf2::Vector3(0.0, 0.0, 0.0), optical_rotation);

    broadcaster.sendTransform(transforms);
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
    publishSensorRigTf();

    geometry_msgs::PoseStamped pose = makeWorldPoseFromNed(*msg);
    gps_pose_pub.publish(pose);

    appendBoundedPath(gps_path, pose);
    gps_path_pub.publish(gps_path);
}

void poseGtCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    updateLatestPoseForDebug(*msg, true);
    publishSensorRigTf();

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
    latest_odom = *msg;
    has_odom = true;

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

bool readImageColor(
    const sensor_msgs::Image& msg,
    uint32_t u,
    uint32_t v,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b)
{
    const bool rgb = msg.encoding == "rgb8" || msg.encoding == "rgba8";
    const bool bgr = msg.encoding == "bgr8" || msg.encoding == "bgra8" || msg.encoding == "8UC3";
    const uint32_t channels = (msg.encoding == "rgba8" || msg.encoding == "bgra8") ? 4 : 3;
    if (!rgb && !bgr) {
        return false;
    }

    const size_t offset = static_cast<size_t>(v) * msg.step + static_cast<size_t>(u) * channels;
    if (offset + 2 >= msg.data.size()) {
        return false;
    }

    if (rgb) {
        r = msg.data[offset];
        g = msg.data[offset + 1];
        b = msg.data[offset + 2];
    } else {
        b = msg.data[offset];
        g = msg.data[offset + 1];
        r = msg.data[offset + 2];
    }
    return true;
}

void publishCameraImagePlane(
    const sensor_msgs::Image::ConstPtr& msg,
    const tf2::Vector3& camera_body_frd,
    bool forward,
    ros::Publisher& publisher,
    ros::Time& last_publish_time)
{
    if (!has_pose_ned || publisher.getNumSubscribers() == 0 || msg->height == 0 || msg->width == 0) {
        return;
    }

    const ros::Time now = ros::Time::now();
    if (!last_publish_time.isZero() && (now - last_publish_time).toSec() < 0.2) {
        return;
    }
    last_publish_time = now;

    const uint32_t sample_step = 12;
    const double plane_distance = 1.0;
    const double half_width = plane_distance * std::tan(M_PI / 6.0);
    const double half_height =
        half_width * static_cast<double>(msg->height) / static_cast<double>(msg->width);
    const double x_dir = forward ? 1.0 : -1.0;
    const double y_dir = forward ? 1.0 : -1.0;
    const uint32_t rows = (msg->height + sample_step - 1) / sample_step;
    const uint32_t cols = (msg->width + sample_step - 1) / sample_step;

    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = kVizFrame;
    cloud.header.stamp = stampOrNow(msg->header.stamp);

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(rows * cols);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud, "b");

    bool unsupported_encoding = false;
    for (uint32_t v = 0; v < msg->height; v += sample_step) {
        for (uint32_t u = 0; u < msg->width; u += sample_step) {
            const double denom_u = std::max<uint32_t>(msg->width - 1, 1);
            const double denom_v = std::max<uint32_t>(msg->height - 1, 1);
            const double image_y =
                (static_cast<double>(u) / denom_u - 0.5) * 2.0 * half_width;
            const double image_z =
                (static_cast<double>(v) / denom_v - 0.5) * 2.0 * half_height;
            const tf2::Vector3 point_body =
                camera_body_frd + tf2::Vector3(x_dir * plane_distance, y_dir * image_y, image_z);
            const geometry_msgs::Point point_world = pointBodyFrdToWorld(point_body, latest_pose_ned);

            uint8_t r = 0;
            uint8_t g = 0;
            uint8_t b = 0;
            if (!readImageColor(*msg, u, v, r, g, b)) {
                unsupported_encoding = true;
            }

            *iter_x = static_cast<float>(point_world.x);
            *iter_y = static_cast<float>(point_world.y);
            *iter_z = static_cast<float>(point_world.z);
            *iter_r = r;
            *iter_g = g;
            *iter_b = b;

            ++iter_x;
            ++iter_y;
            ++iter_z;
            ++iter_r;
            ++iter_g;
            ++iter_b;
        }
    }

    if (unsupported_encoding) {
        ROS_WARN_THROTTLE(5.0, "Unsupported camera image encoding for world image plane: %s", msg->encoding.c_str());
    }

    publisher.publish(cloud);
}

void frontLeftImageCb(const sensor_msgs::Image::ConstPtr& msg)
{
    publishCameraImagePlane(msg, tf2::Vector3(0.175, -0.15, 0.0), true, front_left_image_plane_pub, last_front_left_image_plane);
}

void frontRightImageCb(const sensor_msgs::Image::ConstPtr& msg)
{
    publishCameraImagePlane(msg, tf2::Vector3(0.175, 0.15, 0.0), true, front_right_image_plane_pub, last_front_right_image_plane);
}

void backLeftImageCb(const sensor_msgs::Image::ConstPtr& msg)
{
    publishCameraImagePlane(msg, tf2::Vector3(-0.175, -0.15, 0.0), false, back_left_image_plane_pub, last_back_left_image_plane);
}

void backRightImageCb(const sensor_msgs::Image::ConstPtr& msg)
{
    publishCameraImagePlane(msg, tf2::Vector3(-0.175, 0.15, 0.0), false, back_right_image_plane_pub, last_back_right_image_plane);
}

void lidarCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    if (!has_odom) {
        return;
    }

    sensor_msgs::PointCloud2 out = *msg;
    out.header.frame_id = kVizFrame;
    out.header.stamp = stampOrNow(msg->header.stamp);

    tf2::Quaternion q(
        latest_odom.pose.pose.orientation.x,
        latest_odom.pose.pose.orientation.y,
        latest_odom.pose.pose.orientation.z,
        latest_odom.pose.pose.orientation.w);
    if (q.length2() < 1e-12) {
        return;
    }
    q.normalize();

    tf2::Transform body_to_world_ned(q);
    tf2::Vector3 pos_ned(
        latest_odom.pose.pose.position.x,
        latest_odom.pose.pose.position.y,
        latest_odom.pose.pose.position.z);

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
    front_left_image_plane_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/debug/front_left_image_plane_world", 1);
    front_right_image_plane_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/debug/front_right_image_plane_world", 1);
    back_left_image_plane_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/debug/back_left_image_plane_world", 1);
    back_right_image_plane_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/debug/back_right_image_plane_world", 1);

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
    ros::Subscriber front_left_image_sub =
        nh.subscribe("/airsim_node/drone_1/front_left/Scene", 1, frontLeftImageCb);
    ros::Subscriber front_right_image_sub =
        nh.subscribe("/airsim_node/drone_1/front_right/Scene", 1, frontRightImageCb);
    ros::Subscriber back_left_image_sub =
        nh.subscribe("/airsim_node/drone_1/back_left/Scene", 1, backLeftImageCb);
    ros::Subscriber back_right_image_sub =
        nh.subscribe("/airsim_node/drone_1/back_right/Scene", 1, backRightImageCb);

    ROS_INFO("debug_node started; all debug outputs use frame '%s'", kVizFrame);

    ros::spin();
    return 0;
}
