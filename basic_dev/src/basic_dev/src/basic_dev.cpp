#include <ros/ros.h>
#include <airsim_ros/Takeoff.h>
#include <airsim_ros/VelCmd.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "basic_dev");
    ros::NodeHandle nh;

    ros::ServiceClient takeoff_client = nh.serviceClient<airsim_ros::Takeoff>("/airsim_node/drone_1/takeoff");
    ros::Publisher vel_pub = nh.advertise<airsim_ros::VelCmd>("/airsim_node/drone_1/vel_body_cmd", 10);

    ROS_INFO("[Day0] waiting for takeoff service");
    takeoff_client.waitForExistence();

    ros::Duration(2.0).sleep();

    airsim_ros::Takeoff takeoff_srv;
    takeoff_srv.request.waitOnLastTask = 1;

    if (takeoff_client.call(takeoff_srv)) {
        ROS_INFO("[Day0] takeoff success");
    } else {
        ROS_WARN("[Day0] takeoff failed");
    }

    ros::Rate rate(20);
    ros::Time start_time = ros::Time::now();

    while (ros::ok()) {
        double t = (ros::Time::now() - start_time).toSec();

        airsim_ros::VelCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "";
        cmd.vx = 0.0;
        cmd.vy = 0.0;
        cmd.vz = 0.0;
        cmd.yawRate = 0.0;
        cmd.va = 4;
        cmd.stop = 0;

        if (t > 4.0 && t < 9.0) {
            cmd.vx = 1.0;
        }

        vel_pub.publish(cmd);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}