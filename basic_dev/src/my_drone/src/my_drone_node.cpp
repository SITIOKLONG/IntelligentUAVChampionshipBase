#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <airsim_ros/Takeoff.h>
#include <airsim_ros/VelCmd.h>

#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// ---------- global ----------
geometry_msgs::PoseStamped gps_msg;
geometry_msgs::PoseStamped goal_msg;

bool has_gps = false;
bool has_goal = false;
bool has_lidar = false;

ros::Publisher lidar_world_pub;

void lidarCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    if (!has_gps) return;

    sensor_msgs::PointCloud2 out = *msg;
    out.header.frame_id = "world";
    out.header.stamp = msg->header.stamp;

    Eigen::Vector3d pos(
        gps_msg.pose.position.x,
        gps_msg.pose.position.y,
        gps_msg.pose.position.z
    );

    Eigen::Quaterniond q(
        gps_msg.pose.orientation.w,
        gps_msg.pose.orientation.x,
        gps_msg.pose.orientation.y,
        gps_msg.pose.orientation.z
    );
    q.normalize();

    // MID360 在機體中心上方約 0.05m，NED/FRD 裡上方是 -z
    Eigen::Vector3d lidar_offset_body(0.0, 0.0, -0.05);

    sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        Eigen::Vector3d p_lidar(*iter_x, *iter_y, *iter_z);

        // 先假設 lidar frame = body frame
        Eigen::Vector3d p_body = lidar_offset_body + p_lidar;

        Eigen::Vector3d p_world = pos + q * p_body;

        *iter_x = p_world.x();
        *iter_y = p_world.y();
        *iter_z = p_world.z();
    }

    lidar_world_pub.publish(out);
    has_lidar = true;
}
void gpsCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    gps_msg = *msg;
    has_gps = true;
}

void goalCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    goal_msg = *msg;
    has_goal = true;
}

double clamp(double x, double min_v, double max_v)
{
    if (x < min_v) return min_v;
    if (x > max_v) return max_v;
    return x;
}

double getYawFromQuat(const geometry_msgs::Quaternion& q)
{
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

double distXY(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
    return std::hypot(a.x() - b.x(), a.y() - b.y());
}

Eigen::Vector3d rawToWorld(const Eigen::Vector3d& p)
{
    // 先只看 XY，Z 固定成飛行高度
    double flight_z = -2.2;

    return Eigen::Vector3d(
        p.x(),
        p.y(),
        flight_z
    );
}
std::vector<Eigen::Vector3d> selectSplinePath(
    const std::vector<std::vector<Eigen::Vector3d>>& raw_splines,
    const Eigen::Vector3d& start_w,
    const Eigen::Vector3d& goal_w)
{
    int start_idx = -1;
    int goal_idx = -1;
    double best_start = 1e9;
    double best_goal = 1e9;

    for (int i = 0; i < (int)raw_splines.size(); ++i) {
        if (raw_splines[i].empty()) continue;

        Eigen::Vector3d front = rawToWorld(raw_splines[i].front());

        double ds = distXY(front, start_w);
        double dg = distXY(front, goal_w);

        if (ds < best_start) {
            best_start = ds;
            start_idx = i;
        }

        if (dg < best_goal) {
            best_goal = dg;
            goal_idx = i;
        }
    }

    std::vector<Eigen::Vector3d> path;

    if (start_idx < 0 || goal_idx < 0) {
        ROS_ERROR("Failed to select spline path");
        return path;
    }

    ROS_WARN_STREAM("selected start spline = " << start_idx << ", dist = " << best_start);
    ROS_WARN_STREAM("selected goal spline = " << goal_idx << ", dist = " << best_goal);

    for (const auto& p : raw_splines[start_idx]) {
        path.push_back(rawToWorld(p));
    }

    for (int i = (int)raw_splines[goal_idx].size() - 1; i >= 0; --i) {
        path.push_back(rawToWorld(raw_splines[goal_idx][i]));
    }

    ROS_WARN_STREAM("selected path size = " << path.size());
    return path;
}


void publishSelectedSpline(
    const std::vector<Eigen::Vector3d>& path,
    ros::Publisher& pub)
{
    nav_msgs::Path msg;
    msg.header.frame_id = "world";
    msg.header.stamp = ros::Time::now();

    for (const auto& p : path) {
        geometry_msgs::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = p.z();
        ps.pose.orientation.w = 1.0;
        msg.poses.push_back(ps);
    }

    pub.publish(msg);
}

int findNearestIndex(
    const std::vector<Eigen::Vector3d>& path,
    const Eigen::Vector3d& pos)
{
    int best_idx = 0;
    double best_dist = 1e9;

    for (int i = 0; i < (int)path.size(); ++i) {
        double d = std::hypot(path[i].x() - pos.x(), path[i].y() - pos.y());

        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }

    return best_idx;
}

Eigen::Vector3d getLookaheadPoint(
    const std::vector<Eigen::Vector3d>& path,
    int nearest_idx,
    double lookahead)
{
    double acc = 0.0;

    for (int i = nearest_idx; i + 1 < (int)path.size(); ++i) {
        double ds = std::hypot(
            path[i + 1].x() - path[i].x(),
            path[i + 1].y() - path[i].y()
        );

        acc += ds;

        if (acc >= lookahead) {
            return path[i + 1];
        }
    }

    return path.back();
}

std::vector<std::vector<Eigen::Vector3d>> loadSplines(const std::string& file)
{
    std::ifstream fin(file);
    std::vector<std::vector<Eigen::Vector3d>> paths;

    if (!fin.is_open()) {
        ROS_ERROR_STREAM("Cannot open spline file: " << file);
        return paths;
    }

    std::string line;
    int line_id = 0;

    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        std::vector<double> nums;
        double v;

        while (ss >> v) {
            nums.push_back(v);
        }

        std::vector<Eigen::Vector3d> path;

        for (size_t i = 0; i + 2 < nums.size(); i += 3) {
            path.emplace_back(nums[i], nums[i + 1], nums[i + 2]);
        }

        if (path.size() > 5) {
            paths.push_back(path);

            ROS_WARN_STREAM(
                "spline line " << line_id
                << " points=" << path.size()
                << " first=" << path.front().transpose()
                << " last=" << path.back().transpose()
            );
        }

        line_id++;
    }

    ROS_WARN_STREAM("Loaded spline paths: " << paths.size());
    return paths;
}

void publishAllSplines(
    const std::vector<std::vector<Eigen::Vector3d>>& raw_splines,
    ros::Publisher& pub)
{
    visualization_msgs::MarkerArray arr;

    for (size_t i = 0; i < raw_splines.size(); ++i) {
        visualization_msgs::Marker m;

        m.header.frame_id = "world";
        m.header.stamp = ros::Time::now();

        m.ns = "all_splines";
        m.id = static_cast<int>(i);
        m.type = visualization_msgs::Marker::LINE_STRIP;
        m.action = visualization_msgs::Marker::ADD;

        m.pose.orientation.w = 1.0;

        m.scale.x = 0.4;

        m.color.a = 1.0;
        m.color.r = 0.2;
        m.color.g = 0.8;
        m.color.b = 0.2;

        for (const auto& raw_p : raw_splines[i]) {
            Eigen::Vector3d p = rawToWorld(raw_p);

            geometry_msgs::Point q;
            q.x = p.x();
            q.y = p.y();
            q.z = p.z();

            m.points.push_back(q);
        }

        arr.markers.push_back(m);
    }

    pub.publish(arr);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "my_drone_node");
    ros::NodeHandle nh;

    std::string spline_file = "/basic_dev/src/my_drone/src/Splines.txt";
    std::vector<std::vector<Eigen::Vector3d>> raw_splines =
        loadSplines(spline_file);

    // publish spline
    ros::Publisher all_splines_pub =
    nh.advertise<visualization_msgs::MarkerArray>("/my_drone/all_splines", 1, true);
    ros::Publisher selected_spline_pub =
    nh.advertise<nav_msgs::Path>("/my_drone/selected_spline", 1, true); 
    publishAllSplines(raw_splines, all_splines_pub);
    
    lidar_world_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/my_drone/lidar_world", 1);

    ros::Subscriber gps_sub =
        nh.subscribe("/airsim_node/drone_1/gps", 10, gpsCb);

    ros::Subscriber goal_sub =
        nh.subscribe("/airsim_node/end_goal", 10, goalCb);

    ros::Subscriber lidar_sub =
        nh.subscribe("/airsim_node/drone_1/lidar", 1, lidarCb);

    ros::ServiceClient takeoff_client =
        nh.serviceClient<airsim_ros::Takeoff>("/airsim_node/drone_1/takeoff");

    ros::Publisher vel_pub =
        nh.advertise<airsim_ros::VelCmd>("/airsim_node/drone_1/vel_body_cmd", 10);

    ros::Publisher goal_marker_pub =
        nh.advertise<visualization_msgs::Marker>("/my_drone/end_goal_marker", 1);

    ROS_INFO("[my_drone] waiting for takeoff service...");
    takeoff_client.waitForExistence();

    ros::Duration(1.0).sleep();

    airsim_ros::Takeoff takeoff_srv;
    takeoff_srv.request.waitOnLastTask = 1;

    if (takeoff_client.call(takeoff_srv)) {
        ROS_INFO("[my_drone] takeoff success");
    } else {
        ROS_WARN("[my_drone] takeoff failed, still publishing velocity");
    }

    ros::Rate rate(30);
    ros::Time start_time = ros::Time::now();

    std::vector<Eigen::Vector3d> selected_path;
    bool has_selected_path = false;
    ros::Time last_spline_pub_time = ros::Time::now();

    while (ros::ok()) {
        ros::spinOnce();

        airsim_ros::VelCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "body";

        cmd.vx = 0.0;
        cmd.vy = 0.0;
        cmd.vz = 0.0;
        cmd.yawRate = 0.0;
        cmd.va = 4;
        cmd.stop = 0;

        double t = (ros::Time::now() - start_time).toSec();

if (has_gps && has_goal) {
    Eigen::Vector3d pos(
        gps_msg.pose.position.x,
        gps_msg.pose.position.y,
        gps_msg.pose.position.z
    );

    Eigen::Vector3d goal(
        goal_msg.pose.position.x,
        goal_msg.pose.position.y,
        goal_msg.pose.position.z
    );

    if (!has_selected_path) {
        selected_path = selectSplinePath(raw_splines, pos, goal);
        publishSelectedSpline(selected_path, selected_spline_pub);
        has_selected_path = !selected_path.empty();
    }

    if ((ros::Time::now() - last_spline_pub_time).toSec() > 1.0) {
        publishAllSplines(raw_splines, all_splines_pub);

        if (has_selected_path) {
            publishSelectedSpline(selected_path, selected_spline_pub);
        }

        last_spline_pub_time = ros::Time::now();
    }

    if (has_selected_path) {
        int nearest_idx = findNearestIndex(selected_path, pos);

        double lookahead = 10.0;
        Eigen::Vector3d target =
            getLookaheadPoint(selected_path, nearest_idx, lookahead);

        double dx_w = target.x() - pos.x();
        double dy_w = target.y() - pos.y();

        double yaw = getYawFromQuat(gps_msg.pose.orientation);

        // world velocity direction -> body velocity direction
        double vx_b =  std::cos(yaw) * dx_w + std::sin(yaw) * dy_w;
        double vy_b = -std::sin(yaw) * dx_w + std::cos(yaw) * dy_w;

        double norm = std::sqrt(vx_b * vx_b + vy_b * vy_b);

        double speed = 3.0;

        if (norm > 1e-6) {
            cmd.vx = speed * vx_b / norm;
            cmd.vy = speed * vy_b / norm;
        }

        double target_z = -2.2;
        double ez = target_z - pos.z();

        // 你前面測到上升需要正 vz，所以這裡保留反號
        cmd.vz = -clamp(0.8 * ez, -1.0, 1.0);

        ROS_INFO_THROTTLE(
            0.5,
            "follow spline: nearest=%d target=(%.1f %.1f %.1f) pos=(%.1f %.1f %.1f) cmd=(%.2f %.2f %.2f)",
            nearest_idx,
            target.x(), target.y(), target.z(),
            pos.x(), pos.y(), pos.z(),
            cmd.vx, cmd.vy, cmd.vz
        );
    }


            // ROS_INFO_THROTTLE(
            //     0.5,
            //     "[my_drone] pos=(%.2f %.2f %.2f), goal=(%.2f %.2f), dist=%.2f, yaw=%.2f, cmd=(%.2f %.2f %.2f)",
            //     x, y, z, gx, gy, dist_xy, yaw, cmd.vx, cmd.vy, cmd.vz
            // );

            // publish end goal marker
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "my_drone";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position = goal_msg.pose.position;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 5.0;
            marker.scale.y = 5.0;
            marker.scale.z = 5.0;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            goal_marker_pub.publish(marker);
        } else {
            // 没收到 GPS / goal 前，先慢慢向前，避免 30 秒没离开起点
            if (t > 2.0) {
                cmd.vx = 1.0;
                cmd.vy = 0.0;
                cmd.vz = 0.0;
            }

            // ROS_WARN_THROTTLE(
            //     1.0,
            //     "[my_drone] waiting gps/goal: gps=%d goal=%d",
            //     has_gps,
            //     has_goal
            // );
        }

        vel_pub.publish(cmd);
        rate.sleep();
    }

    return 0;
}