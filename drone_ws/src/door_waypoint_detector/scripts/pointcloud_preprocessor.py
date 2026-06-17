#!/usr/bin/env python3

import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

try:
    import open3d as o3d
except Exception:
    o3d = None


class PointcloudPreprocessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_stereo = None
        self.latest_lidar = None
        self.latest_odom = None
        self.latest_stamp = rospy.Time(0)

        self.stereo_cloud_topic = rospy.get_param("~stereo_cloud_topic", "/stereo_depth/front/points")
        self.lidar_cloud_topic = rospy.get_param("~lidar_cloud_topic", "")
        self.odom_topic = rospy.get_param("~odom_topic", "/eskf_odom")
        self.output_topic = rospy.get_param("~output_topic", "/door_waypoint_detector/preprocessed_cloud_world")
        self.max_input_points = int(rospy.get_param("~max_input_points", 30000))
        self.random_sample_points = int(rospy.get_param("~random_sample_points", 12000))
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.18))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 30.0))
        self.lidar_offset_body = np.asarray(rospy.get_param("~lidar_offset_body", [0.0, 0.0, -0.05]), dtype=np.float64)

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.stereo_sub = rospy.Subscriber(self.stereo_cloud_topic, PointCloud2, self.stereo_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.lidar_sub = None
        if self.lidar_cloud_topic:
            self.lidar_sub = rospy.Subscriber(self.lidar_cloud_topic, PointCloud2, self.lidar_callback, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate_hz, 1.0)), self.timer_callback)
        rospy.loginfo(
            "pointcloud_preprocessor: stereo=%s lidar=%s odom=%s output=%s open3d=%s voxel=%.2f",
            self.stereo_cloud_topic,
            self.lidar_cloud_topic or "disabled",
            self.odom_topic,
            self.output_topic,
            o3d is not None,
            self.voxel_size)

    def stereo_callback(self, msg):
        self.store_cloud(msg, "stereo")

    def lidar_callback(self, msg):
        self.store_cloud(msg, "lidar")

    def odom_callback(self, msg):
        with self.lock:
            self.latest_odom = msg

    def store_cloud(self, msg, source):
        cloud = self.msg_to_numpy(msg)
        if source == "lidar":
            cloud = self.lidar_to_world(cloud)
            if cloud is None:
                return
        with self.lock:
            if source == "stereo":
                self.latest_stereo = cloud
            else:
                self.latest_lidar = cloud
            self.latest_stamp = msg.header.stamp

    def lidar_to_world(self, points_lidar):
        if points_lidar.shape[0] == 0:
            return points_lidar

        with self.lock:
            odom = self.latest_odom

        if odom is None:
            rospy.logwarn_throttle(
                1.0,
                "pointcloud_preprocessor: waiting for odometry on %s before transforming lidar",
                self.odom_topic)
            return None

        q = odom.pose.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return None
        quat /= norm

        pos_ned = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64)

        points_body = points_lidar + self.lidar_offset_body
        points_ned = pos_ned + self.rotate_points(quat, points_body)
        return self.ned_to_world(points_ned)

    def timer_callback(self, _event):
        with self.lock:
            stereo = None if self.latest_stereo is None else self.latest_stereo.copy()
            lidar = None if self.latest_lidar is None else self.latest_lidar.copy()
            stamp = self.latest_stamp

        clouds = [c for c in (stereo, lidar) if c is not None and c.shape[0] > 0]
        if not clouds:
            return
        points = np.vstack(clouds)
        points = self.random_sample(points, self.random_sample_points)
        points = self.voxel_downsample(points)
        self.publish(points, stamp)

    def msg_to_numpy(self, msg):
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            if len(points) >= self.max_input_points:
                break
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def random_sample(points, limit):
        if limit <= 0 or points.shape[0] <= limit:
            return points
        idx = np.random.choice(points.shape[0], size=limit, replace=False)
        return points[idx]

    def voxel_downsample(self, points):
        if points.shape[0] == 0 or self.voxel_size <= 0.0:
            return points
        if o3d is None:
            keys = np.floor(points / self.voxel_size).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            return points[np.sort(idx)]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        down = cloud.voxel_down_sample(self.voxel_size)
        return np.asarray(down.points, dtype=np.float64)

    @staticmethod
    def rotate_points(quat_xyzw, points):
        q_vec = quat_xyzw[:3]
        q_w = quat_xyzw[3]
        uv = np.cross(q_vec, points)
        uuv = np.cross(q_vec, uv)
        return points + 2.0 * (q_w * uv + uuv)

    @staticmethod
    def ned_to_world(points_ned):
        points_world = points_ned.copy()
        points_world[:, 1] *= -1.0
        points_world[:, 2] *= -1.0
        return points_world

    def publish(self, points, stamp):
        header = Header()
        header.frame_id = "world"
        header.stamp = stamp
        self.pub.publish(pc2.create_cloud_xyz32(header, points.astype(np.float32)))


if __name__ == "__main__":
    rospy.init_node("pointcloud_preprocessor")
    PointcloudPreprocessor()
    rospy.spin()
