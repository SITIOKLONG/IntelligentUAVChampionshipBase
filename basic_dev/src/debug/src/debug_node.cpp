#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_broadcaster.h>

ros::Publisher gps_pose_pub;
ros::Publisher gps_path_pub;
nav_msgs::Path gps_path;

void gpsCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped pose = *msg;

    // Foxglove 必須知道這個 pose 屬於哪個座標系
    pose.header.frame_id = "world";

    if (pose.header.stamp.isZero()) {
        pose.header.stamp = ros::Time::now();
    }

    gps_pose_pub.publish(pose);

    // 發 TF: world -> drone_gps
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped tf;

    tf.header.stamp = pose.header.stamp;
    tf.header.frame_id = "world";
    tf.child_frame_id = "drone_gps";

    tf.transform.translation.x = pose.pose.position.x;
    tf.transform.translation.y = pose.pose.position.y;
    tf.transform.translation.z = pose.pose.position.z;
    tf.transform.rotation = pose.pose.orientation;

    br.sendTransform(tf);

    // 順便發飛行軌跡
    gps_path.header.frame_id = "world";
    gps_path.header.stamp = pose.header.stamp;
    gps_path.poses.push_back(pose);

    if (gps_path.poses.size() > 2000) {
        gps_path.poses.erase(gps_path.poses.begin());
    }

    gps_path_pub.publish(gps_path);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "debug_node");
    ros::NodeHandle nh;

    gps_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/debug/gps_pose_world", 10);
    gps_path_pub = nh.advertise<nav_msgs::Path>("/debug/gps_path_world", 10);

    ros::Subscriber gps_sub =
        nh.subscribe("/airsim_node/drone_1/gps", 10, gpsCb);

    ROS_INFO("debug_node started");

    ros::spin();
    return 0;
}