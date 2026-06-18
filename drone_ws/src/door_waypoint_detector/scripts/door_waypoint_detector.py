#!/usr/bin/env python3

import math
import threading

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def ned_to_world(p_ned):
    return np.array([p_ned[0], -p_ned[1], -p_ned[2]], dtype=np.float64)


def make_point(v):
    p = Point()
    p.x = float(v[0])
    p.y = float(v[1])
    p.z = float(v[2])
    return p


def make_color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r = r
    c.g = g
    c.b = b
    c.a = a
    return c


def make_orientation_from_direction(direction):
    x_axis = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(x_axis)
    if norm < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = np.cross(up, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(np.linalg.norm(z_axis), 1e-9)
    rot = np.column_stack((x_axis, y_axis, z_axis))
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-9)
    return q


class DoorTrack:
    def __init__(self, track_id, stamp):
        self.track_id = track_id
        self.left_post = None
        self.right_post = None
        self.center = None
        self.waypoints = []
        self.cloud_points = []
        self.seen_count = 0
        self.last_seen = stamp
        self.last_update = stamp
        self.active = False


class DoorFusionTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.image_topic = rospy.get_param("~image_topic", "/airsim_node/drone_1/front_left/Scene")
        self.preprocessed_cloud_topic = rospy.get_param("~preprocessed_cloud_topic", "/door_waypoint_detector/preprocessed_cloud_world")
        self.detections_topic = rospy.get_param("~detections_topic", "/door_waypoint_detector/yolo_detections")
        self.odom_topic = rospy.get_param("~odom_topic", "/eskf_odom")
        self.annotated_image_topic = rospy.get_param("~annotated_image_topic", "/door_waypoint_detector/debug_image")
        self.marker_topic = rospy.get_param("~marker_topic", "/door_waypoint_detector/markers_world")
        self.yolo_box_cloud_topic = rospy.get_param("~yolo_box_cloud_topic", "/door_waypoint_detector/yolo_box_points_world")
        self.current_door_cloud_topic = rospy.get_param("~current_door_cloud_topic", "/door_waypoint_detector/current_door_points_world")
        self.next_door_cloud_topic = rospy.get_param("~next_door_cloud_topic", "/door_waypoint_detector/next_door_points_world")
        self.corridor_cluster_cloud_topic = rospy.get_param("~corridor_cluster_cloud_topic", "/door_waypoint_detector/corridor_pca_cluster_points_world")
        self.waypoint_path_topic = rospy.get_param("~waypoint_path_topic", "/door_waypoint_detector/waypoints_world")
        self.next_waypoint_topic = rospy.get_param("~next_waypoint_topic", "/door_waypoint_detector/next_waypoint_world")

        self.width = int(rospy.get_param("~image_width", 960))
        self.height = int(rospy.get_param("~image_height", 720))
        self.horizontal_fov_deg = float(rospy.get_param("~horizontal_fov_deg", 60.0))
        self.bbox_margin_px = int(rospy.get_param("~bbox_margin_px", 12))
        self.min_points_in_box = int(rospy.get_param("~min_points_in_box", 8))
        self.post_neighborhood_xy_m = float(rospy.get_param("~post_neighborhood_xy_m", 10.0))
        self.post_neighborhood_z_m = float(rospy.get_param("~post_neighborhood_z_m", 10.0))
        self.post_cluster_radius_m = float(rospy.get_param("~post_cluster_radius_m", 1.2))
        self.post_cluster_max_points = int(rospy.get_param("~post_cluster_max_points", 400))
        self.door_width_m = float(rospy.get_param("~door_width_m", 10.0))
        self.door_width_tolerance_m = float(rospy.get_param("~door_width_tolerance_m", 4.0))
        self.single_door_width_m = float(rospy.get_param("~single_door_width_m", 3.0))
        self.single_door_width_tolerance_m = float(rospy.get_param("~single_door_width_tolerance_m", 2.0))
        self.single_door_depth_m = float(rospy.get_param("~single_door_depth_m", 3.0))
        self.single_door_depth_tolerance_m = float(rospy.get_param("~single_door_depth_tolerance_m", 2.5))
        self.single_door_height_m = float(rospy.get_param("~single_door_height_m", 10.0))
        self.single_door_height_tolerance_m = float(rospy.get_param("~single_door_height_tolerance_m", 5.0))
        self.waypoint_z_offset_m = float(rospy.get_param("~waypoint_z_offset_m", 5.0))
        self.corridor_min_side_points = int(rospy.get_param("~corridor_min_side_points", 2))
        self.corridor_direction_max_change_deg = float(rospy.get_param("~corridor_direction_max_change_deg", 40.0))
        self.corridor_cloud_tube_radius_m = float(rospy.get_param("~corridor_cloud_tube_radius_m", 1.0))
        self.corridor_cloud_search_radius_m = float(rospy.get_param("~corridor_cloud_search_radius_m", 3.0))
        self.corridor_cloud_cluster_radius_m = float(rospy.get_param("~corridor_cloud_cluster_radius_m", 1.0))
        self.corridor_cloud_cluster_max_points = int(rospy.get_param("~corridor_cloud_cluster_max_points", 2000))
        self.corridor_pca_cluster_extend_m = float(rospy.get_param("~corridor_pca_cluster_extend_m", 3.0))
        self.corridor_pca_cluster_include_radius_m = float(rospy.get_param("~corridor_pca_cluster_include_radius_m", 0.6))
        self.corridor_cloud_min_points = int(rospy.get_param("~corridor_cloud_min_points", 12))
        self.corridor_edge_min_distance_m = float(rospy.get_param("~corridor_edge_min_distance_m", 8.0))
        self.corridor_edge_max_distance_m = float(rospy.get_param("~corridor_edge_max_distance_m", 22.0))
        self.corridor_edge_expected_distance_m = float(rospy.get_param("~corridor_edge_expected_distance_m", 15.0))
        self.corridor_edge_max_pca_angle_deg = float(rospy.get_param("~corridor_edge_max_pca_angle_deg", 35.0))
        self.corridor_memory_back_distance_m = float(rospy.get_param(
            "~corridor_memory_back_distance_m",
            self.corridor_edge_expected_distance_m))
        self.corridor_memory_max_age_s = float(rospy.get_param("~corridor_memory_max_age_s", 30.0))
        self.corridor_pca_smoothing_alpha = float(rospy.get_param("~corridor_pca_smoothing_alpha", 0.2))
        self.corridor_lookahead_distance_m = float(rospy.get_param("~corridor_lookahead_distance_m", 6.0))
        self.corridor_center_anchor_alpha = float(rospy.get_param("~corridor_center_anchor_alpha", 0.8))
        self.approach_distance_m = float(rospy.get_param("~approach_distance_m", 2.0))
        self.pass_distance_m = float(rospy.get_param("~pass_distance_m", 2.0))
        self.track_timeout_s = float(rospy.get_param("~track_timeout_s", 8.0))
        self.track_merge_distance_m = float(rospy.get_param("~track_merge_distance_m", 3.0))
        self.min_door_spacing_m = float(rospy.get_param("~min_door_spacing_m", 15.0))
        self.virtual_next_waypoint_distance_m = float(rospy.get_param("~virtual_next_waypoint_distance_m", 15.0))
        self.min_track_seen = int(rospy.get_param("~min_track_seen", 1))
        self.point_refine_radius_m = float(rospy.get_param("~point_refine_radius_m", 0.8))
        self.point_refine_cluster_radius_m = float(rospy.get_param("~point_refine_cluster_radius_m", 0.8))
        self.point_refine_cluster_max_points = int(rospy.get_param("~point_refine_cluster_max_points", 800))
        self.track_cloud_keepalive_radius_m = float(rospy.get_param("~track_cloud_keepalive_radius_m", 4.0))
        self.track_cloud_keepalive_min_points = int(rospy.get_param("~track_cloud_keepalive_min_points", 20))
        self.track_cloud_radius_m = float(rospy.get_param("~track_cloud_radius_m", 2.0))

        self.fx = (self.width * 0.5) / math.tan(math.radians(self.horizontal_fov_deg) * 0.5)
        self.fy = self.fx
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5
        self.front_left_offset_body = np.array([0.175, -0.15, 0.0], dtype=np.float64)

        self.latest_preprocessed_cloud_world = np.empty((0, 3), dtype=np.float64)
        self.latest_odom = None
        self.latest_detections = []
        self.latest_image = None
        self.latest_debug_stamp = rospy.Time(0)
        self.latest_debug_state = ([], None, None, set())
        self.latest_corridor_cluster_points = np.empty((0, 3), dtype=np.float64)
        self.tracked_corridor_clusters = {
            "left": np.empty((0, 3), dtype=np.float64),
            "right": np.empty((0, 3), dtype=np.float64),
        }
        self.tracked_corridor_models = {
            "left": None,
            "right": None,
        }
        self.last_corridor_cluster_update = rospy.Time(0)
        self.tracks = []
        self.next_track_id = 1
        self.corridor_direction = None

        self.image_pub = rospy.Publisher(self.annotated_image_topic, Image, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.yolo_box_cloud_pub = rospy.Publisher(self.yolo_box_cloud_topic, PointCloud2, queue_size=1)
        self.current_door_cloud_pub = rospy.Publisher(self.current_door_cloud_topic, PointCloud2, queue_size=1)
        self.next_door_cloud_pub = rospy.Publisher(self.next_door_cloud_topic, PointCloud2, queue_size=1)
        self.corridor_cluster_cloud_pub = rospy.Publisher(self.corridor_cluster_cloud_topic, PointCloud2, queue_size=1)
        self.path_pub = rospy.Publisher(self.waypoint_path_topic, Path, queue_size=1, latch=True)
        self.next_waypoint_pub = rospy.Publisher(self.next_waypoint_topic, PoseStamped, queue_size=1, latch=True)

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.preprocessed_cloud_sub = rospy.Subscriber(
            self.preprocessed_cloud_topic,
            PointCloud2,
            self.cloud_callback,
            queue_size=1)
        self.detections_sub = rospy.Subscriber(self.detections_topic, Float32MultiArray, self.detections_callback, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.latest_image_callback, queue_size=1, buff_size=2**24)
        self.debug_timer = rospy.Timer(rospy.Duration(0.1), self.debug_timer_callback)

        rospy.loginfo(
            "door_fusion_tracker: image=%s detections=%s cloud=%s odom=%s",
            self.image_topic,
            self.detections_topic,
            self.preprocessed_cloud_topic,
            self.odom_topic)

    def odom_callback(self, msg):
        with self.lock:
            self.latest_odom = msg

    def cloud_callback(self, msg):
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        cloud = np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)
        with self.lock:
            self.latest_preprocessed_cloud_world = cloud

    def detections_callback(self, msg):
        data = list(msg.data)
        if len(data) < 2:
            return
        n = int(data[1])
        detections = []
        offset = 2
        for _ in range(n):
            if offset + 6 > len(data):
                break
            cls, conf, x1, y1, x2, y2 = data[offset:offset + 6]
            offset += 6
            detections.append({
                "class_id": int(cls),
                "conf": float(conf),
                "bbox": np.array([x1, y1, x2, y2], dtype=np.float64),
            })
        stamp = rospy.Time.from_sec(data[0]) if data[0] > 0 else rospy.Time.now()
        with self.lock:
            self.latest_detections = detections
        self.fuse_latest_detections(stamp)

    def latest_image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "door_fusion_tracker: cv_bridge failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_debug_stamp = msg.header.stamp

    def fuse_latest_detections(self, stamp):
        with self.lock:
            cloud_world = self.latest_preprocessed_cloud_world.copy()
            odom = self.latest_odom
            detections = list(self.latest_detections)

        if odom is None or cloud_world.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "door_fusion_tracker: waiting for odom and preprocessed cloud")
            return

        tracks_active_cloud = self.refine_tracks_with_cloud(cloud_world, odom, stamp)
        detections = [det for det in detections if self.valid_post_bbox(det["bbox"])]
        detected_posts, yolo_box_points, candidate_pairs = self.estimate_candidates(detections, cloud_world, odom)
        left_post, right_post = self.select_valid_door_pair(candidate_pairs, odom)

        if left_post is not None and right_post is not None:
            self.update_tracks(left_post, right_post, odom, stamp)

        current_track, next_track = self.current_and_next_tracks(odom, stamp)
        control_waypoints = self.control_waypoints(current_track, next_track)
        if control_waypoints:
            self.publish_path(control_waypoints, stamp)
        self.publish_yolo_box_cloud(yolo_box_points, stamp)
        self.publish_track_clouds(current_track, next_track, cloud_world, stamp)
        with self.lock:
            self.latest_debug_state = (detected_posts, current_track, next_track, tracks_active_cloud)

    def debug_timer_callback(self, _event):
        with self.lock:
            if self.latest_image is None:
                image = None
            else:
                image = self.latest_image.copy()
            stamp = self.latest_debug_stamp if not self.latest_debug_stamp.is_zero() else rospy.Time.now()
            detected_posts, current_track, next_track, tracks_active_cloud = self.latest_debug_state
        if image is None:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.publish_debug(image, stamp, detected_posts, current_track, next_track, tracks_active_cloud)

    def estimate_candidates(self, detections, cloud_world, odom, projected=None, image_pairs=None):
        if projected is None:
            projected = self.project_cloud_to_front_image(cloud_world, odom)
        if image_pairs is None:
            image_pairs = self.select_valid_image_pairs(detections)
        detected_posts = []
        yolo_box_points = []
        candidate_pairs = []
        for left_det, right_det in image_pairs:
            left_post, left_points = self.estimate_post(left_det, projected, cloud_world)
            right_post, right_points = self.estimate_post(right_det, projected, cloud_world)
            yolo_box_points.extend(left_points)
            yolo_box_points.extend(right_points)
            if left_post is None or right_post is None:
                continue
            detected_posts.append((left_det["class_id"], left_det["conf"], left_det["bbox"], left_post))
            detected_posts.append((right_det["class_id"], right_det["conf"], right_det["bbox"], right_post))
            candidate_pairs.append((left_det, right_det, left_post, right_post))
        return detected_posts, yolo_box_points, candidate_pairs

    def project_cloud_to_front_image(self, cloud_world, odom):
        if cloud_world.shape[0] == 0:
            return []
        pos_ned = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64)
        q = np.array([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return []
        q = q / norm
        q_inv = quat_conjugate(q)

        point_ned = cloud_world.copy()
        point_ned[:, 1] *= -1.0
        point_ned[:, 2] *= -1.0
        rel = point_ned - pos_ned

        x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
        qx, qy, qz, qw = q_inv
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        point_body = np.column_stack((
            x + qw * tx + qy * tz - qz * ty,
            y + qw * ty + qz * tx - qx * tz,
            z + qw * tz + qx * ty - qy * tx,
        ))

        point_link = point_body - self.front_left_offset_body
        point_optical = np.column_stack((point_link[:, 1], point_link[:, 2], point_link[:, 0]))
        depth = point_optical[:, 2]
        valid_depth = depth > 0.2
        u = self.fx * point_optical[:, 0] / np.maximum(depth, 1e-6) + self.cx
        v = self.fy * point_optical[:, 1] / np.maximum(depth, 1e-6) + self.cy
        valid = valid_depth & (u >= -50.0) & (u <= self.width + 50.0) & (v >= -50.0) & (v <= self.height + 50.0)
        idx = np.where(valid)[0]
        return [(float(u[i]), float(v[i]), cloud_world[i], float(depth[i])) for i in idx]

    def estimate_post(self, det, projected, cloud_world):
        x1, y1, x2, y2 = det["bbox"]
        x1 -= self.bbox_margin_px
        y1 -= self.bbox_margin_px
        x2 += self.bbox_margin_px
        y2 += self.bbox_margin_px

        box_samples = [(p, depth) for u, v, p, depth in projected if x1 <= u <= x2 and y1 <= v <= y2]
        if len(box_samples) < self.min_points_in_box:
            return None, []

        pts, depths = self.select_foreground_points(box_samples)
        if pts.shape[0] < self.min_points_in_box:
            return None, pts

        seed = np.median(pts, axis=0)
        neighborhood = self.post_neighborhood_points(seed, cloud_world)
        if neighborhood.shape[0] >= self.min_points_in_box:
            if self.valid_single_door_cloud(neighborhood, None):
                return self.robust_post_center(pts), neighborhood
            rospy.logwarn_throttle(
                1.0,
                "door_waypoint_detector: rejected expanded post cloud: %s",
                self.single_door_cloud_reason(neighborhood, None))

        if not self.valid_single_door_cloud(pts, depths):
            rospy.logwarn_throttle(
                1.0,
                "door_waypoint_detector: rejected bbox post cloud: %s",
                self.single_door_cloud_reason(pts, depths))
            return None, pts
        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        keep = pts[dist < np.percentile(dist, 70)]
        if keep.shape[0] >= self.min_points_in_box:
            return np.median(keep, axis=0), keep
        return median, pts

    @staticmethod
    def robust_post_center(pts):
        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        keep = pts[dist < np.percentile(dist, 70)]
        if keep.shape[0] > 0:
            return np.median(keep, axis=0)
        return median

    def post_neighborhood_points(self, seed, cloud_world):
        if cloud_world.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        half_xy = 0.5 * max(
            self.post_neighborhood_xy_m,
            self.single_door_width_m + self.single_door_width_tolerance_m,
            self.single_door_depth_m + self.single_door_depth_tolerance_m)
        max_z_span = max(self.post_neighborhood_z_m, 0.1)
        half_z = 0.5 * max_z_span
        delta = cloud_world - seed
        mask = (
            (np.abs(delta[:, 0]) <= half_xy) &
            (np.abs(delta[:, 1]) <= half_xy) &
            (np.abs(delta[:, 2]) <= half_z)
        )
        points = cloud_world[mask]
        if points.shape[0] < self.min_points_in_box:
            return points
        points = self.cluster_near_seed(points, seed)
        if points.shape[0] < self.min_points_in_box:
            return points
        order = np.argsort(points[:, 2])
        sorted_points = points[order]
        best_start = 0
        best_end = 0
        start = 0
        for end in range(sorted_points.shape[0]):
            while sorted_points[end, 2] - sorted_points[start, 2] > max_z_span:
                start += 1
            if end - start > best_end - best_start:
                best_start = start
                best_end = end
        return sorted_points[best_start:best_end + 1]

    def cluster_near_seed(self, points, seed):
        return self.connected_cluster_near_seed(
            points,
            seed,
            self.post_cluster_radius_m,
            self.post_cluster_max_points)

    def select_foreground_points(self, box_samples):
        pts = np.asarray([p for p, _ in box_samples], dtype=np.float64)
        depths = np.asarray([depth for _, depth in box_samples], dtype=np.float64)
        order = np.argsort(depths)
        pts = pts[order]
        depths = depths[order]

        max_depth_span = self.single_door_depth_m + self.single_door_depth_tolerance_m
        best_start = 0
        best_end = 0
        start = 0
        for end in range(depths.shape[0]):
            while depths[end] - depths[start] > max_depth_span:
                start += 1
            if end - start > best_end - best_start:
                best_start = start
                best_end = end

        selected = pts[best_start:best_end + 1]
        selected_depths = depths[best_start:best_end + 1]
        if selected.shape[0] < self.min_points_in_box:
            return selected, selected_depths

        median = np.median(selected, axis=0)
        dist = np.linalg.norm(selected - median, axis=1)
        keep = dist < np.percentile(dist, 85)
        return selected[keep], selected_depths[keep]

    def valid_single_door_cloud(self, pts, depths):
        return self.single_door_cloud_reason(pts, depths) is None

    def single_door_cloud_reason(self, pts, depths):
        if pts is None or pts.shape[0] == 0:
            return "empty cloud"
        z_span = np.max(pts[:, 2]) - np.min(pts[:, 2])
        if z_span > self.single_door_height_m + self.single_door_height_tolerance_m:
            return "z_span %.2fm > %.2fm" % (
                z_span,
                self.single_door_height_m + self.single_door_height_tolerance_m)

        xy_span = np.ptp(pts[:, :2], axis=0)
        horizontal_span = float(np.max(xy_span))
        if horizontal_span > self.single_door_width_m + self.single_door_width_tolerance_m:
            return "xy_span %.2fm > %.2fm" % (
                horizontal_span,
                self.single_door_width_m + self.single_door_width_tolerance_m)

        if depths is not None and depths.size:
            depth_span = float(np.max(depths) - np.min(depths))
            if depth_span > self.single_door_depth_m + self.single_door_depth_tolerance_m:
                return "depth_span %.2fm > %.2fm" % (
                    depth_span,
                    self.single_door_depth_m + self.single_door_depth_tolerance_m)
        return None

    def valid_post_bbox(self, bbox):
        return True

    def select_valid_image_pairs(self, detections):
        left = [d for d in detections if d["class_id"] == 0]
        right = [d for d in detections if d["class_id"] == 1]
        pairs = []
        for left_det in left:
            for right_det in right:
                if self.valid_image_pair(left_det["bbox"], right_det["bbox"]):
                    pairs.append((left_det, right_det))
        return pairs

    def select_valid_door_pair(self, candidate_pairs, odom):
        if not candidate_pairs:
            return None, None

        drone_world = ned_to_world(np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64))

        best_pair = None
        best_score = None
        best_rejected_width = None
        best_rejected_error = None
        best_rejected_xy_width = None
        best_rejected_dz = None
        for left_det, right_det, left_post, right_post in candidate_pairs:
            center = 0.5 * (left_post + right_post)
            post_delta = right_post - left_post
            width = np.linalg.norm(post_delta[:2])
            width_error = abs(width - self.door_width_m)
            z_diff = abs(float(post_delta[2]))
            if width_error > self.door_width_tolerance_m:
                if best_rejected_error is None or width_error < best_rejected_error:
                    best_rejected_width = width
                    best_rejected_error = width_error
                    best_rejected_xy_width = np.linalg.norm(post_delta[:2])
                    best_rejected_dz = z_diff
                continue

            distance = np.linalg.norm(center - drone_world)
            confidence = 0.5 * (left_det["conf"] + right_det["conf"])
            score = width_error + 0.02 * distance - confidence
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (left_post, right_post)

        if best_pair is None:
            rospy.logwarn_throttle(
                1.0,
                "door_waypoint_detector: no valid 3D door pair from %d 2D-matched pairs; best_width=%.2fm xy=%.2fm dz=%.2fm error=%.2fm allowed_error=%.2fm",
                len(candidate_pairs),
                best_rejected_width if best_rejected_width is not None else -1.0,
                best_rejected_xy_width if best_rejected_xy_width is not None else -1.0,
                best_rejected_dz if best_rejected_dz is not None else -1.0,
                best_rejected_error if best_rejected_error is not None else -1.0,
                self.door_width_tolerance_m)
            return None, None
        return best_pair

    def valid_image_pair(self, left_bbox, right_bbox):
        lx1, ly1, lx2, ly2 = left_bbox
        rx1, ry1, rx2, ry2 = right_bbox
        lw = max(1.0, lx2 - lx1)
        rw = max(1.0, rx2 - rx1)
        lh = max(1.0, ly2 - ly1)
        rh = max(1.0, ry2 - ry1)
        left_cx = 0.5 * (lx1 + lx2)
        right_cx = 0.5 * (rx1 + rx2)
        left_cy = 0.5 * (ly1 + ly2)
        right_cy = 0.5 * (ry1 + ry2)

        if left_cx >= right_cx:
            return False
        if right_cx - left_cx < 0.5 * (lw + rw):
            return False
        return True

    def update_tracks(self, left_post, right_post, odom, stamp, guessed_center=None, guessed_waypoints=None):
        stamp_sec = stamp.to_sec() if not stamp.is_zero() else rospy.Time.now().to_sec()
        if guessed_center is not None:
            center = guessed_center
        elif left_post is not None and right_post is not None:
            center = 0.5 * (left_post + right_post)
        elif left_post is not None:
            center = left_post
        elif right_post is not None:
            center = right_post
        else:
            return

        track = self.find_track(center)
        if track is None:
            nearest_dist = self.nearest_track_distance(center)
            if nearest_dist is not None and nearest_dist < self.min_door_spacing_m:
                rospy.logwarn_throttle(
                    1.0,
                    "door_waypoint_detector: reject new door track %.1fm from existing track; min spacing %.1fm",
                    nearest_dist,
                    self.min_door_spacing_m)
                return
            track = DoorTrack(self.next_track_id, stamp_sec)
            self.next_track_id += 1
            self.tracks.append(track)

        alpha = 0.35
        if left_post is not None:
            track.left_post = left_post if track.left_post is None else (1.0 - alpha) * track.left_post + alpha * left_post
        if right_post is not None:
            track.right_post = right_post if track.right_post is None else (1.0 - alpha) * track.right_post + alpha * right_post

        if track.left_post is not None and track.right_post is not None:
            track.center = 0.5 * (track.left_post + track.right_post)
            track.waypoints = self.make_waypoints(track, odom)
        elif guessed_center is not None:
            track.center = guessed_center
            track.waypoints = guessed_waypoints or []
        elif track.center is None:
            track.center = center

        track.seen_count += 1
        track.last_seen = stamp_sec
        track.last_update = stamp_sec
        track.active = True
        self.prune_tracks(stamp_sec)

    def find_track(self, center):
        best = None
        best_dist = self.track_merge_distance_m
        for track in self.tracks:
            if track.center is None:
                continue
            dist = np.linalg.norm(track.center - center)
            if dist < best_dist:
                best = track
                best_dist = dist
        return best

    def nearest_track_distance(self, center):
        distances = [
            np.linalg.norm(track.center - center)
            for track in self.tracks
            if track.center is not None
        ]
        if not distances:
            return None
        return min(distances)

    def prune_tracks(self, stamp_sec):
        self.tracks = [
            t for t in self.tracks
            if t.center is not None and stamp_sec - t.last_seen <= self.track_timeout_s
        ]

    def valid_tracks(self, stamp):
        stamp_sec = stamp.to_sec() if not stamp.is_zero() else rospy.Time.now().to_sec()
        self.prune_tracks(stamp_sec)
        return [
            t for t in self.tracks
            if t.center is not None and t.seen_count >= self.min_track_seen
        ]

    def current_and_next_tracks(self, odom, stamp):
        candidates = self.valid_tracks(stamp)
        if not candidates:
            return None, None
        drone_world = ned_to_world(np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64))
        forward_world = self.estimate_corridor_direction(odom, stamp)
        ordered_chain = self.ordered_corridor_tracks(stamp)
        if len(ordered_chain) > 1:
            chain_items = []
            for idx, track in enumerate(ordered_chain):
                rel = track.center - drone_world
                forward_dist = float(np.dot(rel, forward_world))
                lateral_dist = float(np.linalg.norm(rel - forward_dist * forward_world))
                chain_items.append((idx, forward_dist, lateral_dist, track))
            ahead_items = [item for item in chain_items if item[1] > -2.0]
            source = ahead_items if ahead_items else chain_items
            idx, _forward_dist, _lateral_dist, current = min(
                source,
                key=lambda item: (max(0.0, item[1]), item[2]))
            next_track = ordered_chain[idx + 1] if idx + 1 < len(ordered_chain) else None
            return current, next_track
        ahead = []
        behind = []
        for track in candidates:
            rel = track.center - drone_world
            forward_dist = float(np.dot(rel, forward_world))
            lateral_dist = float(np.linalg.norm(rel - forward_dist * forward_world))
            item = (forward_dist, lateral_dist, track)
            if forward_dist > -2.0:
                ahead.append(item)
            else:
                behind.append(item)
        ordered_items = sorted(ahead, key=lambda item: (max(0.0, item[0]), item[1]))
        if len(ordered_items) < 2:
            ordered_items.extend(sorted(behind, key=lambda item: (abs(item[0]), item[1])))
        ordered = [item[2] for item in ordered_items]
        current = ordered[0]
        next_track = ordered[1] if len(ordered) > 1 else None
        return current, next_track

    @staticmethod
    def drone_forward_world(odom):
        q = np.array([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = q / norm
        qx, qy, qz, qw = q
        # Rotate NED/body x-forward unit vector by q, then convert NED vector to world.
        tx = 0.0
        ty = 2.0 * qz
        tz = -2.0 * qy
        forward_ned = np.array([
            1.0 + qy * tz - qz * ty,
            qw * ty + qz * tx - qx * tz,
            qw * tz + qx * ty - qy * tx,
        ], dtype=np.float64)
        forward_world = np.array([forward_ned[0], -forward_ned[1], -forward_ned[2]], dtype=np.float64)
        forward_world[2] = 0.0
        norm = np.linalg.norm(forward_world)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return forward_world / norm

    def refine_tracks_with_cloud(self, cloud_world, odom, stamp):
        stamp_sec = stamp.to_sec() if not stamp.is_zero() else rospy.Time.now().to_sec()
        active = []
        for track in self.tracks:
            changed = False
            if track.left_post is not None:
                refined = self.refine_point_from_cloud(track.left_post, cloud_world)
                if refined is not None:
                    track.left_post = 0.75 * track.left_post + 0.25 * refined
                    changed = True
            if track.right_post is not None:
                refined = self.refine_point_from_cloud(track.right_post, cloud_world)
                if refined is not None:
                    track.right_post = 0.75 * track.right_post + 0.25 * refined
                    changed = True
            cloud_alive = self.track_has_cloud_support(track, cloud_world)
            if (changed or cloud_alive) and track.left_post is not None and track.right_post is not None:
                track.center = 0.5 * (track.left_post + track.right_post)
                track.waypoints = self.make_waypoints(track, odom)
                track.cloud_points = self.cloud_points_for_track(track, cloud_world)
                track.last_seen = stamp_sec
                track.last_update = stamp_sec
                active.append(track.track_id)
        return set(active)

    def refine_point_from_cloud(self, point, cloud_world):
        if cloud_world.shape[0] == 0:
            return None
        dist = np.linalg.norm(cloud_world - point, axis=1)
        near = cloud_world[dist < self.point_refine_radius_m]
        if near.shape[0] < self.min_points_in_box:
            return None
        cluster = self.connected_cluster_near_seed(
            near,
            point,
            self.point_refine_cluster_radius_m,
            self.point_refine_cluster_max_points)
        if cluster.shape[0] < self.min_points_in_box:
            return None
        return np.median(cluster, axis=0)

    def track_has_cloud_support(self, track, cloud_world):
        anchors = [p for p in (track.left_post, track.right_post, track.center) if p is not None]
        if not anchors or cloud_world.shape[0] == 0:
            return False
        for anchor in anchors:
            dist = np.linalg.norm(cloud_world - anchor, axis=1)
            if int(np.count_nonzero(dist < self.track_cloud_keepalive_radius_m)) >= self.track_cloud_keepalive_min_points:
                return True
        return False

    @staticmethod
    def connected_cluster_near_seed(points, seed, radius, max_points):
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        if points.shape[0] > max_points:
            dist = np.linalg.norm(points - seed, axis=1)
            points = points[np.argsort(dist)[:max_points]]
        dist_to_seed = np.linalg.norm(points - seed, axis=1)
        start_idx = int(np.argmin(dist_to_seed))
        radius_sq = radius * radius
        visited = np.zeros(points.shape[0], dtype=bool)
        cluster = []
        queue = [start_idx]
        visited[start_idx] = True
        while queue:
            idx = queue.pop()
            cluster.append(idx)
            delta = points - points[idx]
            near_idx = np.where(np.einsum("ij,ij->i", delta, delta) <= radius_sq)[0]
            for next_idx in near_idx:
                if not visited[next_idx]:
                    visited[next_idx] = True
                    queue.append(int(next_idx))
        return points[np.asarray(cluster, dtype=np.int64)]

    @staticmethod
    def pca_direction(points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] < 2:
            return None
        centered = points - np.mean(points, axis=0)
        cov = centered.T @ centered / max(points.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        return direction / norm

    def estimate_corridor_direction(self, odom, stamp):
        tracks = [
            t for t in self.valid_tracks(stamp)
            if t.left_post is not None and t.right_post is not None
        ]
        left_points = [t.left_post for t in tracks]
        right_points = [t.right_post for t in tracks]
        directions = []
        left_dir = self.pca_direction(left_points) if len(left_points) >= self.corridor_min_side_points else None
        right_dir = self.pca_direction(right_points) if len(right_points) >= self.corridor_min_side_points else None
        if left_dir is not None:
            directions.append(left_dir)
        if right_dir is not None:
            if directions and float(np.dot(right_dir, directions[0])) < 0.0:
                right_dir = -right_dir
            directions.append(right_dir)

        if not directions and self.corridor_direction is not None:
            return self.corridor_direction
        if not directions:
            direction = self.drone_forward_world(odom)
            return direction

        direction = np.mean(np.vstack(directions), axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = directions[0]
        else:
            direction = direction / norm

        forward = self.drone_forward_world(odom)
        if float(np.dot(direction, forward)) < 0.0:
            direction = -direction

        if self.corridor_direction is not None:
            old = self.corridor_direction
            if float(np.dot(direction, old)) < 0.0:
                direction = -direction
            cos_limit = math.cos(math.radians(self.corridor_direction_max_change_deg))
            if float(np.dot(direction, old)) < cos_limit:
                direction = old

        self.corridor_direction = direction / max(np.linalg.norm(direction), 1e-9)
        return self.corridor_direction

    def ordered_corridor_tracks(self, stamp):
        tracks = [
            t for t in self.valid_tracks(stamp)
            if t.center is not None
        ]
        if len(tracks) <= 1:
            return tracks
        cloud_world = self.latest_preprocessed_cloud_world.copy()
        if cloud_world.shape[0] > 0:
            edges = self.corridor_cloud_edges(tracks, cloud_world)
            ordered = self.order_tracks_from_edges(tracks, edges)
            if len(ordered) > 1:
                return ordered
        direction = self.corridor_direction
        if direction is None:
            centers = [t.center for t in tracks]
            direction = self.pca_direction(centers)
        if direction is None:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        origin = np.mean(np.vstack([t.center for t in tracks]), axis=0)
        return sorted(tracks, key=lambda t: float(np.dot(t.center - origin, direction)))

    @staticmethod
    def normalized_vector(vector):
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return None
        return vector / norm

    @staticmethod
    def align_direction(direction, reference):
        if reference is None:
            return direction
        return -direction if float(np.dot(direction, reference)) < 0.0 else direction

    def corridor_cloud_edges(self, tracks, cloud_world):
        candidates = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a = tracks[i]
                b = tracks[j]
                if a.center is None or b.center is None:
                    continue
                center_dist = float(np.linalg.norm(b.center - a.center))
                if center_dist < self.corridor_edge_min_distance_m or center_dist > self.corridor_edge_max_distance_m:
                    continue
                left = self.side_corridor_cloud_evidence(a, b, "left", cloud_world)
                right = self.side_corridor_cloud_evidence(a, b, "right", cloud_world)
                if left is None and right is None:
                    continue
                point_count = 0
                align_score = 0.0
                if left is not None:
                    point_count += left["count"]
                    align_score += left["alignment"]
                if right is not None:
                    point_count += right["count"]
                    align_score += right["alignment"]
                distance_penalty = abs(center_dist - self.corridor_edge_expected_distance_m)
                score = point_count + 20.0 * align_score - 2.0 * distance_penalty
                candidates.append({
                    "a": a,
                    "b": b,
                    "left": left,
                    "right": right,
                    "score": score,
                    "distance": center_dist,
                })

        parent = {track: track for track in tracks}
        degree = {track: 0 for track in tracks}

        def find(track):
            while parent[track] is not track:
                parent[track] = parent[parent[track]]
                track = parent[track]
            return track

        selected = []
        for edge in sorted(candidates, key=lambda item: item["score"], reverse=True):
            a = edge["a"]
            b = edge["b"]
            if degree[a] >= 2 or degree[b] >= 2:
                continue
            root_a = find(a)
            root_b = find(b)
            if root_a is root_b:
                continue
            parent[root_b] = root_a
            degree[a] += 1
            degree[b] += 1
            selected.append(edge)
        return selected

    def order_tracks_from_edges(self, tracks, edges):
        if not edges:
            return []
        adjacency = {track: [] for track in tracks}
        for edge in edges:
            adjacency[edge["a"]].append((edge["b"], edge))
            adjacency[edge["b"]].append((edge["a"], edge))
        used_edges = set()
        components = []
        for start in tracks:
            if not adjacency[start]:
                continue
            if any(start in comp for comp in components):
                continue
            endpoints = [track for track in tracks if adjacency[track] and len(adjacency[track]) == 1]
            start_node = start
            for endpoint in endpoints:
                if self.same_edge_component(endpoint, start, adjacency):
                    start_node = endpoint
                    break
            ordered = []
            prev = None
            node = start_node
            while node is not None:
                ordered.append(node)
                next_node = None
                best_score = None
                for neighbor, edge in adjacency[node]:
                    edge_id = id(edge)
                    if neighbor is prev or edge_id in used_edges:
                        continue
                    if best_score is None or edge["score"] > best_score:
                        next_node = neighbor
                        best_score = edge["score"]
                if next_node is None:
                    break
                for neighbor, edge in adjacency[node]:
                    if neighbor is next_node:
                        used_edges.add(id(edge))
                        break
                prev = node
                node = next_node
            components.append(ordered)
        if not components:
            return []
        ordered = max(components, key=len)
        if self.latest_odom is not None and len(ordered) >= 2:
            reference = self.corridor_direction
            if reference is None:
                reference = self.drone_forward_world(self.latest_odom)
            if float(np.dot(ordered[-1].center - ordered[0].center, reference)) < 0.0:
                ordered = list(reversed(ordered))
        remaining = [track for track in tracks if track not in ordered]
        if remaining:
            direction = self.corridor_direction
            if direction is None and len(ordered) >= 2:
                direction = self.normalized_vector(ordered[-1].center - ordered[0].center)
            if direction is None:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            origin = ordered[0].center if ordered else np.mean(np.vstack([t.center for t in tracks]), axis=0)
            remaining = sorted(remaining, key=lambda t: float(np.dot(t.center - origin, direction)))
            ordered.extend(remaining)
        return ordered

    @staticmethod
    def same_edge_component(a, b, adjacency):
        stack = [a]
        seen = set()
        while stack:
            node = stack.pop()
            if node is b:
                return True
            if node in seen:
                continue
            seen.add(node)
            for neighbor, _edge in adjacency[node]:
                if neighbor not in seen:
                    stack.append(neighbor)
        return False

    def smooth_segment_directions(self, ordered, odom):
        if len(ordered) < 2:
            fallback = self.corridor_direction
            if fallback is None:
                fallback = self.drone_forward_world(odom)
            return []

        reference = self.corridor_direction
        if reference is None:
            reference = self.drone_forward_world(odom)
        reference = reference / max(np.linalg.norm(reference), 1e-9)
        cos_limit = math.cos(math.radians(self.corridor_direction_max_change_deg))

        smoothed = []
        previous = None
        for i in range(len(ordered) - 1):
            raw = self.normalized_vector(ordered[i + 1].center - ordered[i].center)
            if raw is None:
                raw = previous if previous is not None else reference
            raw = self.align_direction(raw, previous if previous is not None else reference)
            if previous is not None and float(np.dot(raw, previous)) < cos_limit:
                raw = previous
            smoothed.append(raw)
            previous = raw
        return smoothed

    def local_corridor_direction(self, track, odom, stamp):
        ordered = self.ordered_corridor_tracks(stamp)
        if track not in ordered:
            return self.estimate_corridor_direction(odom, stamp)
        idx = ordered.index(track)
        segment_dirs = self.smooth_segment_directions(ordered, odom)
        directions = []
        if segment_dirs:
            if idx == 0:
                directions.append(segment_dirs[0])
            elif idx >= len(segment_dirs):
                directions.append(segment_dirs[-1])
            else:
                directions.append(segment_dirs[idx - 1])
                directions.append(segment_dirs[idx])

        if 0 < idx < len(ordered) - 1:
            prev_track = ordered[idx - 1]
            next_track = ordered[idx + 1]
            if prev_track.left_post is not None and next_track.left_post is not None:
                directions.append(next_track.left_post - prev_track.left_post)
            if prev_track.right_post is not None and next_track.right_post is not None:
                directions.append(next_track.right_post - prev_track.right_post)
        elif idx + 1 < len(ordered):
            next_track = ordered[idx + 1]
            if track.left_post is not None and next_track.left_post is not None:
                directions.append(next_track.left_post - track.left_post)
            if track.right_post is not None and next_track.right_post is not None:
                directions.append(next_track.right_post - track.right_post)
        elif idx > 0:
            prev_track = ordered[idx - 1]
            if track.left_post is not None and prev_track.left_post is not None:
                directions.append(track.left_post - prev_track.left_post)
            if track.right_post is not None and prev_track.right_post is not None:
                directions.append(track.right_post - prev_track.right_post)

        valid = []
        for direction in directions:
            normalized = self.normalized_vector(direction)
            if normalized is not None:
                valid.append(normalized)
        if not valid:
            return self.estimate_corridor_direction(odom, stamp)
        base = valid[0]
        aligned = [base]
        cos_limit = math.cos(math.radians(self.corridor_direction_max_change_deg))
        for direction in valid[1:]:
            direction = self.align_direction(direction, base)
            if float(np.dot(direction, base)) >= cos_limit:
                aligned.append(direction)
        direction = np.mean(np.vstack(aligned), axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        reference = self.corridor_direction
        if reference is None:
            reference = self.drone_forward_world(odom)
        direction = self.align_direction(direction, reference)
        return direction

    def corridor_lateral_axis(self, direction):
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = np.cross(up, direction)
        norm = np.linalg.norm(lateral)
        if norm < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return lateral / norm

    def corridor_half_width(self, track, lateral):
        if track.left_post is not None and track.right_post is not None:
            center = track.center
            left_offset = abs(float(np.dot(track.left_post - center, lateral)))
            right_offset = abs(float(np.dot(track.right_post - center, lateral)))
            half_width = 0.5 * (left_offset + right_offset)
            if half_width > 1e-3:
                return half_width
        return 0.5 * self.door_width_m

    def corridor_marker_points(self, ordered, stamp):
        odom = self.latest_odom
        if odom is None:
            return [], [], []
        cloud_world = self.latest_preprocessed_cloud_world.copy()
        if cloud_world.shape[0] > 0:
            segments = self.corridor_cloud_marker_segments(ordered, cloud_world)
            if any(segments):
                return segments
        center_points = []
        left_points = []
        right_points = []
        previous_lateral = None
        for track in ordered:
            if track.center is None:
                continue
            direction = self.local_corridor_direction(track, odom, stamp)
            lateral = self.corridor_lateral_axis(direction)
            if previous_lateral is not None and float(np.dot(lateral, previous_lateral)) < 0.0:
                lateral = -lateral
            previous_lateral = lateral
            half_width = self.corridor_half_width(track, lateral)
            center_points.append(track.center)
            left_points.append(track.center + lateral * half_width)
            right_points.append(track.center - lateral * half_width)
        return center_points, left_points, right_points

    def corridor_cloud_marker_segments(self, ordered, cloud_world):
        center_segments = []
        left_segments = []
        right_segments = []
        cluster_debug_points = []
        edges = self.corridor_cloud_edges(ordered, cloud_world)
        if not edges:
            tracked_segments = self.tracked_corridor_marker_segments()
            self.latest_corridor_cluster_points = self.tracked_corridor_cluster_points()
            return tracked_segments
        for edge in edges:
            left_segment = edge["left"]["segment"] if edge["left"] is not None else None
            right_segment = edge["right"]["segment"] if edge["right"] is not None else None
            if left_segment is not None:
                left_segment = self.extended_marker_segment_from_evidence(edge["left"])
                left_segments.extend(left_segment)
                cluster_debug_points.append(edge["left"]["cluster"])
            if right_segment is not None:
                right_segment = self.extended_marker_segment_from_evidence(edge["right"])
                right_segments.extend(right_segment)
                cluster_debug_points.append(edge["right"]["cluster"])
            if left_segment is not None and right_segment is not None:
                center_segments.extend([
                    0.5 * (left_segment[0] + right_segment[0]),
                    0.5 * (left_segment[1] + right_segment[1]),
                ])
        if cluster_debug_points:
            tracked = self.tracked_corridor_cluster_points()
            if tracked.shape[0] > 0:
                cluster_debug_points.append(tracked)
            self.latest_corridor_cluster_points = np.vstack(cluster_debug_points)
        else:
            self.latest_corridor_cluster_points = self.tracked_corridor_cluster_points()
        return center_segments, left_segments, right_segments

    def tracked_corridor_marker_segments(self):
        left_cluster = self.tracked_corridor_clusters["left"]
        right_cluster = self.tracked_corridor_clusters["right"]
        if left_cluster.shape[0] < self.corridor_cloud_min_points or right_cluster.shape[0] < self.corridor_cloud_min_points:
            return [], [], []
        left_anchor = np.mean(left_cluster, axis=0)
        right_anchor = np.mean(right_cluster, axis=0)
        reference = self.corridor_direction if self.corridor_direction is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        left_model = self.extended_side_segment_from_cluster(
            left_cluster,
            left_anchor,
            reference,
            self.virtual_next_waypoint_distance_m)
        right_model = self.extended_side_segment_from_cluster(
            right_cluster,
            right_anchor,
            reference,
            self.virtual_next_waypoint_distance_m)
        if left_model is None or right_model is None:
            return [], [], []
        left_points = [left_model["start"], left_model["end"]]
        right_points = [right_model["start"], right_model["end"]]
        center_points = [
            0.5 * (left_model["start"] + right_model["start"]),
            0.5 * (left_model["end"] + right_model["end"]),
        ]
        return center_points, left_points, right_points

    def extended_marker_segment_from_evidence(self, evidence):
        segment = evidence["segment"]
        reference = segment[1] - segment[0]
        model = self.extended_side_segment_from_cluster(
            evidence["cluster"],
            segment[0],
            reference,
            self.virtual_next_waypoint_distance_m)
        if model is None:
            return segment
        return [model["start"], model["end"]]

    def side_corridor_cloud_segment(self, current, next_track, side, cloud_world):
        p0 = current.left_post if side == "left" else current.right_post
        p1 = next_track.left_post if side == "left" else next_track.right_post
        if p0 is None or p1 is None:
            return None
        points = self.points_near_segment(cloud_world, p0, p1, self.corridor_cloud_tube_radius_m)
        if points.shape[0] < self.corridor_cloud_min_points:
            return [p0, p1]
        segment = self.pca_segment_from_points(points, p1 - p0)
        return segment if segment is not None else [p0, p1]

    def side_corridor_cloud_evidence(self, current, next_track, side, cloud_world):
        p0 = current.left_post if side == "left" else current.right_post
        p1 = next_track.left_post if side == "left" else next_track.right_post
        if p0 is None or p1 is None:
            return None
        reference = p1 - p0
        reference_norm = np.linalg.norm(reference)
        if reference_norm < 1e-9:
            return None
        points, distance_to_segment = self.points_near_segment_with_distances(
            cloud_world,
            p0,
            p1,
            max(self.corridor_cloud_search_radius_m, self.corridor_cloud_tube_radius_m))
        if points.shape[0] < self.corridor_cloud_min_points:
            return None

        reference = reference / reference_norm
        min_alignment = math.cos(math.radians(self.corridor_edge_max_pca_angle_deg))

        if points.shape[0] > self.corridor_cloud_cluster_max_points:
            order = np.argsort(distance_to_segment)
            points = points[order[:self.corridor_cloud_cluster_max_points]]

        best = self.best_corridor_side_cluster(points, reference, min_alignment)
        if best is None:
            return None

        cluster, direction, alignment = best
        refined_cluster = self.refine_corridor_cluster_along_pca(cluster, cloud_world, reference)
        if refined_cluster.shape[0] >= cluster.shape[0]:
            refined_direction = self.pca_direction(refined_cluster)
            if refined_direction is not None:
                refined_alignment = abs(float(np.dot(refined_direction, reference)))
                if refined_alignment >= min_alignment:
                    cluster = refined_cluster
                    direction = refined_direction
                    alignment = refined_alignment
        segment = self.pca_segment_from_points(cluster, p1 - p0)
        if segment is None:
            segment = [p0, p1]
        return {
            "segment": segment,
            "cluster": cluster,
            "count": int(cluster.shape[0]),
            "alignment": alignment,
        }

    def refine_corridor_cluster_along_pca(self, cluster, cloud_world, reference):
        if cluster.shape[0] < self.corridor_cloud_min_points or cloud_world.shape[0] == 0:
            return cluster
        direction = self.pca_direction(cluster)
        if direction is None:
            return cluster
        direction[2] = 0.0
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return cluster
        direction = self.align_direction(direction / norm, reference)

        centroid = np.mean(cluster, axis=0)
        projection = np.dot(cluster - centroid, direction)
        start = centroid + direction * (float(np.min(projection)) - self.corridor_pca_cluster_extend_m)
        end = centroid + direction * (float(np.max(projection)) + self.corridor_pca_cluster_extend_m)
        extra, _distance_to_segment = self.points_near_segment_with_distances(
            cloud_world,
            start,
            end,
            self.corridor_pca_cluster_include_radius_m)
        if extra.shape[0] == 0:
            return cluster

        merged = np.unique(np.vstack((cluster, extra)), axis=0)
        merged = self.connected_cluster_near_seed(
            merged,
            centroid,
            self.corridor_cloud_cluster_radius_m,
            self.corridor_cloud_cluster_max_points)
        if merged.shape[0] > self.corridor_cloud_cluster_max_points:
            order = np.argsort(np.linalg.norm(merged - centroid, axis=1))
            merged = merged[order[:self.corridor_cloud_cluster_max_points]]
        return merged

    @staticmethod
    def points_near_segment(cloud_world, p0, p1, radius):
        points, _distances = DoorFusionTracker.points_near_segment_with_distances(cloud_world, p0, p1, radius)
        return points

    @staticmethod
    def points_near_segment_with_distances(cloud_world, p0, p1, radius):
        if cloud_world.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
        direction = p1 - p0
        length_sq = float(np.dot(direction, direction))
        if length_sq < 1e-9:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
        rel = cloud_world - p0
        t = np.dot(rel, direction) / length_sq
        mask_t = (t >= 0.0) & (t <= 1.0)
        closest = p0 + np.outer(np.clip(t, 0.0, 1.0), direction)
        dist = np.linalg.norm(cloud_world - closest, axis=1)
        mask = mask_t & (dist <= radius)
        return cloud_world[mask], dist[mask]

    def best_corridor_side_cluster(self, points, reference, min_alignment):
        clusters = self.euclidean_clusters(points, self.corridor_cloud_cluster_radius_m)
        best = None
        best_score = None
        for cluster in clusters:
            if cluster.shape[0] < self.corridor_cloud_min_points:
                continue
            direction = self.pca_direction(cluster)
            if direction is None:
                continue
            alignment = abs(float(np.dot(direction, reference)))
            if alignment < min_alignment:
                continue
            projection = np.dot(cluster - np.mean(cluster, axis=0), direction)
            span = float(np.max(projection) - np.min(projection)) if projection.size else 0.0
            score = float(cluster.shape[0]) + 20.0 * alignment + 0.5 * span
            if best_score is None or score > best_score:
                best = (cluster, direction, alignment)
                best_score = score
        return best

    @staticmethod
    def euclidean_clusters(points, radius):
        if points.shape[0] == 0:
            return []
        radius_sq = radius * radius
        visited = np.zeros(points.shape[0], dtype=bool)
        clusters = []
        for start_idx in range(points.shape[0]):
            if visited[start_idx]:
                continue
            cluster = []
            queue = [start_idx]
            visited[start_idx] = True
            while queue:
                idx = queue.pop()
                cluster.append(idx)
                delta = points - points[idx]
                near_idx = np.where(np.einsum("ij,ij->i", delta, delta) <= radius_sq)[0]
                for next_idx in near_idx:
                    if not visited[next_idx]:
                        visited[next_idx] = True
                        queue.append(int(next_idx))
            clusters.append(points[np.asarray(cluster, dtype=np.int64)])
        return clusters

    @staticmethod
    def pca_segment_from_points(points, reference_direction):
        direction = DoorFusionTracker.pca_direction(points)
        if direction is None:
            return None
        if float(np.dot(direction, reference_direction)) < 0.0:
            direction = -direction
        centroid = np.mean(points, axis=0)
        projection = np.dot(points - centroid, direction)
        start = centroid + direction * float(np.min(projection))
        end = centroid + direction * float(np.max(projection))
        return [start, end]

    def make_waypoints(self, track, odom):
        if track.center is None:
            return []
        direction = self.local_corridor_direction(track, odom, rospy.Time.now())
        return [self.waypoint_from_track(track, direction)]

    def waypoint_from_track(self, track, direction):
        waypoint = track.center.copy()
        waypoint[2] += self.waypoint_z_offset_m
        return waypoint

    def corridor_centerline_from_side_pca(self, current_track, next_track, odom, stamp):
        if current_track is None or current_track.center is None:
            return []
        cloud_world = self.latest_preprocessed_cloud_world.copy()

        ordered = self.ordered_corridor_tracks(stamp)
        edges = self.corridor_cloud_edges(ordered, cloud_world) if cloud_world.shape[0] > 0 else []
        edge = self.select_corridor_edge_for_control(edges, current_track, next_track)

        base_direction = self.local_corridor_direction(current_track, odom, stamp)
        if edge is not None:
            edge_direction = edge["b"].center - edge["a"].center
            edge_direction[2] = 0.0
            if np.linalg.norm(edge_direction) > 1e-9:
                base_direction = self.align_direction(edge_direction / np.linalg.norm(edge_direction), base_direction)
            self.update_tracked_corridor_clusters(edge, odom, base_direction, stamp)
        else:
            self.prune_tracked_corridor_clusters(odom, base_direction, stamp)

        left_cluster = self.tracked_corridor_clusters["left"]
        right_cluster = self.tracked_corridor_clusters["right"]
        if left_cluster.shape[0] < self.corridor_cloud_min_points or right_cluster.shape[0] < self.corridor_cloud_min_points:
            return []

        left_anchor = current_track.left_post if current_track.left_post is not None else current_track.center
        right_anchor = current_track.right_post if current_track.right_post is not None else current_track.center
        extension = self.pass_distance_m if next_track is not None else self.virtual_next_waypoint_distance_m
        left_model = self.extended_side_segment_from_cluster(left_cluster, left_anchor, base_direction, extension)
        right_model = self.extended_side_segment_from_cluster(right_cluster, right_anchor, base_direction, extension)
        if left_model is None or right_model is None:
            return []
        left_model = self.smooth_corridor_side_model("left", left_model, base_direction)
        right_model = self.smooth_corridor_side_model("right", right_model, base_direction)

        forward = np.mean(np.vstack([left_model["direction"], right_model["direction"]]), axis=0)
        forward[2] = 0.0
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            return []
        forward = self.align_direction(forward / forward_norm, base_direction)
        forward = self.anchor_direction_to_door_chain(forward, current_track, next_track)

        start = 0.5 * (left_model["start"] + right_model["start"])
        end = 0.5 * (left_model["end"] + right_model["end"])
        start[2] = current_track.center[2] + self.waypoint_z_offset_m
        end[2] = current_track.center[2] + self.waypoint_z_offset_m
        start, end = self.anchor_centerline_to_door_chain(start, end, forward, current_track, next_track)
        return self.lookahead_centerline_waypoints(start, end, forward, odom, next_track is not None)

    def anchor_direction_to_door_chain(self, forward, current_track, next_track):
        if next_track is None or next_track.center is None:
            return forward
        door_direction = np.array(next_track.center - current_track.center, dtype=np.float64, copy=True)
        door_direction[2] = 0.0
        norm = np.linalg.norm(door_direction)
        if norm < 1e-9:
            return forward
        door_direction = self.align_direction(door_direction / norm, forward)
        alpha = min(max(self.corridor_center_anchor_alpha, 0.0), 1.0)
        anchored = (1.0 - alpha) * forward + alpha * door_direction
        anchored[2] = 0.0
        anchored_norm = np.linalg.norm(anchored)
        if anchored_norm < 1e-9:
            return forward
        return self.align_direction(anchored / anchored_norm, forward)

    def anchor_centerline_to_door_chain(self, start, end, forward, current_track, next_track):
        anchors = [current_track.center]
        if next_track is not None and next_track.center is not None:
            anchors.append(next_track.center)
        door_center = np.mean(np.vstack(anchors), axis=0)
        door_center[2] += self.waypoint_z_offset_m

        line_origin = 0.5 * (start + end)
        projection = float(np.dot(door_center - line_origin, forward))
        closest = line_origin + forward * projection
        correction = door_center - closest
        correction -= forward * float(np.dot(correction, forward))
        correction[2] = 0.0
        alpha = min(max(self.corridor_center_anchor_alpha, 0.0), 1.0)
        start = start + alpha * correction
        end = end + alpha * correction

        return start, end

    def lookahead_centerline_waypoints(self, start, end, forward, odom, has_next_track):
        span = float(np.dot(end - start, forward))
        if span < 0.5:
            return [(end, forward)]
        drone_world = ned_to_world(np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64))
        drone_projection = float(np.dot(drone_world - start, forward))
        lookahead_projection = max(0.0, drone_projection) + self.corridor_lookahead_distance_m
        if has_next_track:
            lookahead_projection = min(lookahead_projection, span)
        target = start + forward * lookahead_projection
        target[2] = start[2]
        if has_next_track and np.linalg.norm(end - target) < 0.5:
            target = end
        return [(target, forward)]

    def smooth_corridor_side_model(self, side, model, reference_direction):
        previous = self.tracked_corridor_models.get(side)
        if previous is None:
            self.tracked_corridor_models[side] = model
            return model

        alpha = min(max(self.corridor_pca_smoothing_alpha, 0.0), 1.0)
        direction = self.align_direction(model["direction"], previous["direction"])
        smoothed_direction = (1.0 - alpha) * previous["direction"] + alpha * direction
        smoothed_direction[2] = 0.0
        norm = np.linalg.norm(smoothed_direction)
        if norm < 1e-9:
            smoothed_direction = self.align_direction(direction, reference_direction)
        else:
            smoothed_direction = self.align_direction(smoothed_direction / norm, reference_direction)

        previous_center = 0.5 * (previous["start"] + previous["end"])
        current_center = 0.5 * (model["start"] + model["end"])
        center = (1.0 - alpha) * previous_center + alpha * current_center
        previous_length = float(np.linalg.norm(previous["end"] - previous["start"]))
        current_length = float(np.linalg.norm(model["end"] - model["start"]))
        length = max((1.0 - alpha) * previous_length + alpha * current_length, 0.5)
        smoothed = {
            "start": center - smoothed_direction * (0.5 * length),
            "end": center + smoothed_direction * (0.5 * length),
            "direction": smoothed_direction,
            "cluster": model.get("cluster", np.empty((0, 3), dtype=np.float64)),
        }
        self.tracked_corridor_models[side] = smoothed
        return smoothed

    def update_tracked_corridor_clusters(self, edge, odom, direction, stamp):
        for side in ("left", "right"):
            evidence = edge.get(side)
            if evidence is None or evidence.get("cluster") is None:
                continue
            cluster = evidence["cluster"]
            if cluster.shape[0] == 0:
                continue
            existing = self.tracked_corridor_clusters[side]
            merged = cluster if existing.shape[0] == 0 else np.vstack((existing, cluster))
            self.tracked_corridor_clusters[side] = self.pruned_corridor_memory_points(merged, odom, direction)
        self.last_corridor_cluster_update = stamp

    def prune_tracked_corridor_clusters(self, odom, direction, stamp):
        if self.last_corridor_cluster_update.is_zero():
            return
        age = (stamp - self.last_corridor_cluster_update).to_sec()
        if self.corridor_memory_max_age_s > 0.0 and age > self.corridor_memory_max_age_s:
            self.tracked_corridor_clusters["left"] = np.empty((0, 3), dtype=np.float64)
            self.tracked_corridor_clusters["right"] = np.empty((0, 3), dtype=np.float64)
            self.tracked_corridor_models["left"] = None
            self.tracked_corridor_models["right"] = None
            return
        for side in ("left", "right"):
            self.tracked_corridor_clusters[side] = self.pruned_corridor_memory_points(
                self.tracked_corridor_clusters[side],
                odom,
                direction)
            if self.tracked_corridor_clusters[side].shape[0] < self.corridor_cloud_min_points:
                self.tracked_corridor_models[side] = None

    def pruned_corridor_memory_points(self, points, odom, direction):
        if points.shape[0] == 0:
            return points
        drone_world = ned_to_world(np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ], dtype=np.float64))
        direction = np.array(direction, dtype=np.float64, copy=True)
        direction[2] = 0.0
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return points
        direction = direction / norm
        forward = np.dot(points - drone_world, direction)
        keep = forward >= -self.corridor_memory_back_distance_m
        kept = points[keep]
        if kept.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        kept = np.unique(kept, axis=0)
        centroid = np.mean(kept, axis=0)
        kept = self.connected_cluster_near_seed(
            kept,
            centroid,
            self.corridor_cloud_cluster_radius_m,
            self.corridor_cloud_cluster_max_points)
        if kept.shape[0] > self.corridor_cloud_cluster_max_points:
            order = np.argsort(np.abs(np.dot(kept - drone_world, direction)))
            kept = kept[order[:self.corridor_cloud_cluster_max_points]]
        return kept

    def tracked_corridor_cluster_points(self):
        clusters = [
            cluster for cluster in self.tracked_corridor_clusters.values()
            if cluster.shape[0] > 0
        ]
        if not clusters:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(clusters)

    @staticmethod
    def select_corridor_edge_for_control(edges, current_track, next_track):
        candidates = [
            edge for edge in edges
            if edge["left"] is not None and edge["right"] is not None
        ]
        if not candidates:
            return None
        if current_track is not None and next_track is not None:
            for edge in candidates:
                if ((edge["a"] is current_track and edge["b"] is next_track) or
                        (edge["a"] is next_track and edge["b"] is current_track)):
                    return edge
        if current_track is not None:
            current_edges = [
                edge for edge in candidates
                if edge["a"] is current_track or edge["b"] is current_track
            ]
            if current_edges:
                return max(current_edges, key=lambda item: item["score"])
        return max(candidates, key=lambda item: item["score"])

    def extended_side_segment_from_cluster(self, cluster, anchor, reference_direction, extension):
        direction = self.pca_direction(cluster)
        if direction is None:
            return None
        direction[2] = 0.0
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        direction = self.align_direction(direction / norm, reference_direction)

        centroid = np.mean(cluster, axis=0)
        anchor_projection = float(np.dot(anchor - centroid, direction))
        cluster_projection = np.dot(cluster - centroid, direction)
        forward_span = max(float(np.max(cluster_projection) - anchor_projection), 0.0)
        start = centroid + direction * anchor_projection
        end = start + direction * (forward_span + max(extension, 0.0))
        return {
            "start": start,
            "end": end,
            "direction": direction,
            "cluster": cluster,
        }

    def control_waypoints(self, current_track, next_track):
        waypoints = []
        odom = self.latest_odom
        if odom is None:
            return waypoints
        stamp = rospy.Time.now()
        self.estimate_corridor_direction(odom, stamp)
        pca_centerline = self.corridor_centerline_from_side_pca(current_track, next_track, odom, stamp)
        if pca_centerline:
            return pca_centerline

        if current_track is not None and current_track.center is not None:
            direction = self.local_corridor_direction(current_track, odom, stamp)
            waypoints.append((self.waypoint_from_track(current_track, direction), direction))
        if next_track is not None and next_track.center is not None:
            direction = self.local_corridor_direction(next_track, odom, stamp)
            next_waypoint = self.waypoint_from_track(next_track, direction)
            last_point = waypoints[-1][0] if waypoints else None
            if last_point is None or np.linalg.norm(next_waypoint - last_point) > 0.5:
                waypoints.append((next_waypoint, direction))
        return waypoints

    def publish_path(self, waypoints, stamp):
        path = Path()
        path.header.frame_id = "world"
        path.header.stamp = stamp
        for item in waypoints:
            if isinstance(item, tuple):
                point, direction = item
            else:
                point = item
                direction = self.corridor_direction if self.corridor_direction is not None else np.array([1.0, 0.0, 0.0])
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = make_point(point)
            q = make_orientation_from_direction(direction)
            pose.pose.orientation.x = float(q[0])
            pose.pose.orientation.y = float(q[1])
            pose.pose.orientation.z = float(q[2])
            pose.pose.orientation.w = float(q[3])
            path.poses.append(pose)
        self.path_pub.publish(path)
        if path.poses:
            self.next_waypoint_pub.publish(path.poses[0])

    def publish_yolo_box_cloud(self, points, stamp):
        header = Header()
        header.frame_id = "world"
        header.stamp = stamp
        if points:
            cloud = pc2.create_cloud_xyz32(header, np.asarray(points, dtype=np.float32))
        else:
            cloud = pc2.create_cloud_xyz32(header, [])
        self.yolo_box_cloud_pub.publish(cloud)

    def publish_track_clouds(self, current_track, next_track, cloud_world, stamp):
        self.publish_track_cloud(self.current_door_cloud_pub, current_track, cloud_world, stamp)
        self.publish_track_cloud(self.next_door_cloud_pub, next_track, cloud_world, stamp)

    def publish_track_cloud(self, publisher, track, cloud_world, stamp):
        header = Header()
        header.frame_id = "world"
        header.stamp = stamp
        points = self.cloud_points_for_track(track, cloud_world) if track is not None else np.empty((0, 3), dtype=np.float64)
        if points.shape[0] > 0:
            cloud = pc2.create_cloud_xyz32(header, points.astype(np.float32))
        else:
            cloud = pc2.create_cloud_xyz32(header, [])
        publisher.publish(cloud)

    def cloud_points_for_track(self, track, cloud_world):
        if track is None or track.center is None:
            return np.empty((0, 3), dtype=np.float64)
        if cloud_world.shape[0] == 0:
            if len(track.cloud_points) > 0:
                return np.asarray(track.cloud_points, dtype=np.float64)
            return np.empty((0, 3), dtype=np.float64)
        anchors = [track.center]
        if track.left_post is not None:
            anchors.append(track.left_post)
        if track.right_post is not None:
            anchors.append(track.right_post)
        mask = np.zeros(cloud_world.shape[0], dtype=bool)
        half_xy = 0.5 * self.post_neighborhood_xy_m
        half_z = 0.5 * self.post_neighborhood_z_m
        for anchor in anchors:
            delta = cloud_world - anchor
            mask |= (
                (np.abs(delta[:, 0]) <= half_xy) &
                (np.abs(delta[:, 1]) <= half_xy) &
                (np.abs(delta[:, 2]) <= half_z)
            )
        points = cloud_world[mask]
        if points.shape[0] > 0:
            track.cloud_points = points
            return points
        if len(track.cloud_points) > 0:
            return np.asarray(track.cloud_points, dtype=np.float64)
        return np.empty((0, 3), dtype=np.float64)

    def publish_debug(self, image, stamp, detected_posts, current_track, next_track, cloud_refined_track_ids):
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = "world"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 1
        stamp_sec = stamp.to_sec() if not stamp.is_zero() else rospy.Time.now().to_sec()
        for cls, conf, bbox, post in detected_posts:
            color = make_color(0.1, 0.8, 1.0) if cls == 0 else make_color(1.0, 0.4, 0.1)
            name = "left_post" if cls == 0 else "right_post"
            markers.markers.append(self.sphere_marker(marker_id, name, post, 0.35, color, stamp))
            marker_id += 1
            cv_color = (255, 180, 30) if cls == 0 else (30, 120, 255)
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), cv_color, 2)
            cv2.putText(image, f"{name} {conf:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)

        for track in self.tracks:
            if track.center is None:
                continue
            age = max(0.0, stamp_sec - track.last_seen)
            fresh = age < 0.7
            refined = track.track_id in cloud_refined_track_ids
            if track is current_track:
                color = make_color(0.2, 1.0, 0.2, 1.0 if fresh else 0.65)
            elif track is next_track:
                color = make_color(1.0, 0.75, 0.1, 0.9 if fresh else 0.55)
            else:
                color = make_color(0.6, 0.6, 0.6, 0.45)
            if refined and not fresh:
                color = make_color(0.4, 0.7, 1.0, 0.65)
            scale = 0.55 if track is current_track else 0.45 if track is next_track else 0.32
            markers.markers.append(self.sphere_marker(marker_id, f"tracked_door_{track.track_id}", track.center, scale, color, stamp))
            marker_id += 1
            markers.markers.append(self.text_marker(marker_id, track, age, stamp))
            marker_id += 1

            if track.left_post is not None:
                markers.markers.append(self.sphere_marker(marker_id, f"tracked_left_{track.track_id}", track.left_post, 0.24, make_color(0.1, 0.8, 1.0, color.a), stamp))
                marker_id += 1
            if track.right_post is not None:
                markers.markers.append(self.sphere_marker(marker_id, f"tracked_right_{track.track_id}", track.right_post, 0.24, make_color(1.0, 0.4, 0.1, color.a), stamp))
                marker_id += 1

        marker_id = self.append_corridor_markers(markers, marker_id, stamp)

        current_waypoints = current_track.waypoints if current_track is not None else []
        next_waypoints = next_track.waypoints if next_track is not None else []
        if current_waypoints:
            for i, point in enumerate(current_waypoints):
                markers.markers.append(self.sphere_marker(marker_id, f"current_waypoint_{i}", point, 0.28, make_color(0.2, 1.0, 0.2), stamp))
                marker_id += 1
        if next_waypoints:
            for i, point in enumerate(next_waypoints):
                markers.markers.append(self.sphere_marker(marker_id, f"next_waypoint_{i}", point, 0.24, make_color(1.0, 0.75, 0.1), stamp))
                marker_id += 1

        marker_id = self.append_waypoint_line(markers, marker_id, "current_door_waypoints", current_waypoints, make_color(0.2, 1.0, 0.2), stamp)
        self.append_waypoint_line(markers, marker_id, "next_door_waypoints", next_waypoints, make_color(1.0, 0.75, 0.1), stamp)

        self.marker_pub.publish(markers)
        out = self.bridge.cv2_to_imgmsg(image, "bgr8")
        out.header.frame_id = "front_left"
        out.header.stamp = stamp
        self.image_pub.publish(out)

    def append_corridor_markers(self, markers, marker_id, stamp):
        ordered = self.ordered_corridor_tracks(stamp)
        if len(ordered) < 2:
            self.latest_corridor_cluster_points = np.empty((0, 3), dtype=np.float64)
            self.publish_corridor_cluster_cloud(stamp)
            return marker_id
        center_points, left_points, right_points = self.corridor_marker_points(ordered, stamp)
        segment_specs = [
            ("corridor_center_segments", center_points, make_color(0.7, 0.2, 1.0, 0.9), 0.14),
            ("corridor_left_parallel", left_points, make_color(0.1, 0.8, 1.0, 0.65), 0.08),
            ("corridor_right_parallel", right_points, make_color(1.0, 0.4, 0.1, 0.65), 0.08),
        ]
        for namespace, points, color, width in segment_specs:
            if len(points) >= 2:
                marker_id = self.append_line_list_marker(markers, marker_id, namespace, points, color, width, stamp)
        self.publish_corridor_cluster_cloud(stamp)
        return marker_id

    def publish_corridor_cluster_cloud(self, stamp):
        header = Header()
        header.frame_id = "world"
        header.stamp = stamp
        points = self.latest_corridor_cluster_points
        if points.shape[0] > 0:
            cloud = pc2.create_cloud_xyz32(header, points.astype(np.float32))
        else:
            cloud = pc2.create_cloud_xyz32(header, [])
        self.corridor_cluster_cloud_pub.publish(cloud)

    def append_line_list_marker(self, markers, marker_id, namespace, points, color, width, stamp):
        line = Marker()
        line.header.frame_id = "world"
        line.header.stamp = stamp
        line.ns = namespace
        line.id = marker_id
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.scale.x = width
        line.color = color
        line.points = [make_point(p) for p in points]
        markers.markers.append(line)
        return marker_id + 1

    def append_polyline_marker(self, markers, marker_id, namespace, points, color, width, stamp):
        line = Marker()
        line.header.frame_id = "world"
        line.header.stamp = stamp
        line.ns = namespace
        line.id = marker_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = width
        line.color = color
        line.points = [make_point(p) for p in points]
        markers.markers.append(line)
        return marker_id + 1

    def append_waypoint_line(self, markers, marker_id, namespace, waypoints, color, stamp):
        if len(waypoints) < 2:
            return marker_id
        line = Marker()
        line.header.frame_id = "world"
        line.header.stamp = stamp
        line.ns = namespace
        line.id = marker_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.color = color
        line.points = [make_point(p) for p in waypoints]
        markers.markers.append(line)
        return marker_id + 1

    def sphere_marker(self, marker_id, namespace, point, scale, color, stamp):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = make_point(point)
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        marker.lifetime = rospy.Duration(0.5)
        return marker

    def text_marker(self, marker_id, track, age, stamp):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = stamp
        marker.ns = "door_track_text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = make_point(track.center + np.array([0.0, 0.0, 0.8], dtype=np.float64))
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.45
        marker.color = make_color(1.0, 1.0, 1.0, 0.85)
        marker.text = f"door {track.track_id} seen={track.seen_count} age={age:.1f}s"
        marker.lifetime = rospy.Duration(0.5)
        return marker


DoorWaypointDetector = DoorFusionTracker


if __name__ == "__main__":
    rospy.init_node("door_fusion_tracker")
    DoorFusionTracker()
    rospy.spin()
