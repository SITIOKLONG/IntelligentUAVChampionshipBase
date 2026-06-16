#include <ros/ros.h>

#include <airsim_ros/Takeoff.h>
#include <airsim_ros/VelCmd.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <std_srvs/Empty.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace
{
constexpr const char* kControlFrame = "control_map";

geometry_msgs::Twist latest_manual_cmd;
ros::Time latest_manual_cmd_time;
bool has_manual_cmd = false;

std::vector<geometry_msgs::PoseStamped> waypoints;
nav_msgs::Path waypoint_path;
ros::Publisher waypoint_pub;

double clampValue(double value, double min_value, double max_value)
{
    return std::max(min_value, std::min(max_value, value));
}

class KeyboardReader
{
public:
    KeyboardReader() : enabled_(isatty(STDIN_FILENO) == 1)
    {
        if (!enabled_) {
            return;
        }

        tcgetattr(STDIN_FILENO, &old_termios_);
        termios raw = old_termios_;
        raw.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }

    ~KeyboardReader()
    {
        if (enabled_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
        }
    }

    bool enabled() const
    {
        return enabled_;
    }

    bool read(char& key)
    {
        if (!enabled_) {
            return false;
        }
        return ::read(STDIN_FILENO, &key, 1) == 1;
    }

private:
    bool enabled_;
    termios old_termios_;
};

void printKeyboardHelp()
{
    ROS_INFO(
        "[manual] keys: w/s x forward/back, a/d y left/right, r/f z up/down, "
        "q/e yaw left/right, +/- speed, space hover, x hard stop");
}

void fillHoverCommand(airsim_ros::VelCmd& cmd, bool hard_stop)
{
    cmd.vx = 0.0;
    cmd.vy = 0.0;
    cmd.vz = 0.0;
    cmd.yawRate = 0.0;
    cmd.va = 4;
    cmd.stop = hard_stop ? 1 : 0;
}

void manualCmdCb(const geometry_msgs::Twist::ConstPtr& msg)
{
    latest_manual_cmd = *msg;
    latest_manual_cmd_time = ros::Time::now();
    has_manual_cmd = true;
}

void publishWaypoints()
{
    waypoint_path.header.stamp = ros::Time::now();
    waypoint_path.header.frame_id = kControlFrame;
    waypoint_path.poses = waypoints;
    waypoint_pub.publish(waypoint_path);
}

void addWaypointCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped wp = *msg;
    wp.header.stamp = ros::Time::now();
    if (wp.header.frame_id.empty()) {
        wp.header.frame_id = kControlFrame;
    }
    if (std::abs(wp.pose.orientation.x) < 1e-12 &&
        std::abs(wp.pose.orientation.y) < 1e-12 &&
        std::abs(wp.pose.orientation.z) < 1e-12 &&
        std::abs(wp.pose.orientation.w) < 1e-12) {
        wp.pose.orientation.w = 1.0;
    }

    waypoints.push_back(wp);
    publishWaypoints();
    ROS_INFO(
        "[manual] added waypoint #%zu: x=%.2f y=%.2f z=%.2f",
        waypoints.size(),
        wp.pose.position.x,
        wp.pose.position.y,
        wp.pose.position.z);
}

bool clearWaypointsCb(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
    waypoints.clear();
    publishWaypoints();
    ROS_INFO("[manual] cleared waypoints");
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "my_drone_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    double speed = 1.5;
    double yaw_rate = 25.0;
    double speed_step = 0.5;
    double max_speed = 6.0;
    double manual_cmd_timeout = 0.5;
    bool auto_takeoff = true;

    pnh.param("speed", speed, speed);
    pnh.param("yaw_rate", yaw_rate, yaw_rate);
    pnh.param("speed_step", speed_step, speed_step);
    pnh.param("max_speed", max_speed, max_speed);
    pnh.param("manual_cmd_timeout", manual_cmd_timeout, manual_cmd_timeout);
    pnh.param("auto_takeoff", auto_takeoff, auto_takeoff);

    ros::Subscriber manual_cmd_sub =
        nh.subscribe("/my_drone/manual_cmd", 10, manualCmdCb);
    ros::Subscriber add_waypoint_sub =
        nh.subscribe("/my_drone/add_waypoint", 10, addWaypointCb);
    ros::ServiceServer clear_waypoints_srv =
        nh.advertiseService("/my_drone/clear_waypoints", clearWaypointsCb);

    waypoint_pub =
        nh.advertise<nav_msgs::Path>("/my_drone/waypoints", 1, true);
    ros::Publisher vel_pub =
        nh.advertise<airsim_ros::VelCmd>("/airsim_node/drone_1/vel_body_cmd", 10);

    if (auto_takeoff) {
        ros::ServiceClient takeoff_client =
            nh.serviceClient<airsim_ros::Takeoff>("/airsim_node/drone_1/takeoff");
        ROS_INFO("[manual] waiting for takeoff service...");
        takeoff_client.waitForExistence();

        airsim_ros::Takeoff takeoff_srv;
        takeoff_srv.request.waitOnLastTask = 0;
        if (takeoff_client.call(takeoff_srv)) {
            ROS_INFO("[manual] takeoff command sent");
        } else {
            ROS_WARN("[manual] takeoff service call failed; continuing");
        }
    }

    KeyboardReader keyboard;
    if (keyboard.enabled()) {
        printKeyboardHelp();
    } else {
        ROS_WARN("[manual] no TTY keyboard; use /my_drone/manual_cmd Twist topic");
    }

    publishWaypoints();

    airsim_ros::VelCmd keyboard_cmd;
    fillHoverCommand(keyboard_cmd, false);

    ros::Rate rate(30);
    while (ros::ok()) {
        ros::spinOnce();

        char key = 0;
        while (keyboard.read(key)) {
            key = static_cast<char>(
                std::tolower(static_cast<unsigned char>(key)));

            fillHoverCommand(keyboard_cmd, false);
            keyboard_cmd.va = 5;

            if (key == 'w') {
                keyboard_cmd.vx = speed;
            } else if (key == 's') {
                keyboard_cmd.vx = -speed;
            } else if (key == 'a') {
                keyboard_cmd.vy = -speed;
            } else if (key == 'd') {
                keyboard_cmd.vy = speed;
            } else if (key == 'r') {
                keyboard_cmd.vz = speed;
            } else if (key == 'f') {
                keyboard_cmd.vz = -speed;
            } else if (key == 'q') {
                keyboard_cmd.yawRate = -yaw_rate;
            } else if (key == 'e') {
                keyboard_cmd.yawRate = yaw_rate;
            } else if (key == '+') {
                speed = clampValue(speed + speed_step, 0.0, max_speed);
                ROS_INFO("[manual] speed %.2f", speed);
            } else if (key == '-') {
                speed = clampValue(speed - speed_step, 0.0, max_speed);
                ROS_INFO("[manual] speed %.2f", speed);
            } else if (key == ' ') {
                fillHoverCommand(keyboard_cmd, false);
            } else if (key == 'x') {
                fillHoverCommand(keyboard_cmd, true);
            } else if (key == 'h') {
                printKeyboardHelp();
            }
        }

        airsim_ros::VelCmd cmd = keyboard_cmd;
        if (has_manual_cmd &&
            (ros::Time::now() - latest_manual_cmd_time).toSec() < manual_cmd_timeout) {
            cmd.header.stamp = ros::Time::now();
            cmd.header.frame_id = "body";
            cmd.vx = latest_manual_cmd.linear.x;
            cmd.vy = latest_manual_cmd.linear.y;
            cmd.vz = latest_manual_cmd.linear.z;
            cmd.yawRate = latest_manual_cmd.angular.z;
            cmd.va = 5;
            cmd.stop = 0;
        } else {
            cmd.header.stamp = ros::Time::now();
            cmd.header.frame_id = "body";
 q        }

        vel_pub.publish(cmd);
        rate.sleep();
    }

    return 0;
}
