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
        self.waypoint_path_topic = rospy.get_param("~waypoint_path_topic", "/door_waypoint_detector/waypoints_world")
        self.next_waypoint_topic = rospy.get_param("~next_waypoint_topic", "/door_waypoint_detector/next_waypoint_world")

        self.width = int(rospy.get_param("~image_width", 960))
        self.height = int(rospy.get_param("~image_height", 720))
        self.horizontal_fov_deg = float(rospy.get_param("~horizontal_fov_deg", 60.0))
        self.bbox_margin_px = int(rospy.get_param("~bbox_margin_px", 12))
        self.min_points_in_box = int(rospy.get_param("~min_points_in_box", 8))
        self.post_neighborhood_xy_m = float(rospy.get_param("~post_neighborhood_xy_m", 10.0))
        self.post_neighborhood_z_m = float(rospy.get_param("~post_neighborhood_z_m", 10.0))
        self.door_width_m = float(rospy.get_param("~door_width_m", 10.0))
        self.door_width_tolerance_m = float(rospy.get_param("~door_width_tolerance_m", 4.0))
        self.single_door_width_m = float(rospy.get_param("~single_door_width_m", 3.0))
        self.single_door_width_tolerance_m = float(rospy.get_param("~single_door_width_tolerance_m", 2.0))
        self.single_door_depth_m = float(rospy.get_param("~single_door_depth_m", 3.0))
        self.single_door_depth_tolerance_m = float(rospy.get_param("~single_door_depth_tolerance_m", 2.5))
        self.single_door_height_m = float(rospy.get_param("~single_door_height_m", 10.0))
        self.single_door_height_tolerance_m = float(rospy.get_param("~single_door_height_tolerance_m", 5.0))
        self.waypoint_z_offset_m = float(rospy.get_param("~waypoint_z_offset_m", 5.0))
        self.approach_distance_m = float(rospy.get_param("~approach_distance_m", 2.0))
        self.pass_distance_m = float(rospy.get_param("~pass_distance_m", 2.0))
        self.track_timeout_s = float(rospy.get_param("~track_timeout_s", 8.0))
        self.track_merge_distance_m = float(rospy.get_param("~track_merge_distance_m", 3.0))
        self.min_door_spacing_m = float(rospy.get_param("~min_door_spacing_m", 15.0))
        self.min_track_seen = int(rospy.get_param("~min_track_seen", 1))
        self.point_refine_radius_m = float(rospy.get_param("~point_refine_radius_m", 0.8))
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
        self.tracks = []
        self.next_track_id = 1

        self.image_pub = rospy.Publisher(self.annotated_image_topic, Image, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.yolo_box_cloud_pub = rospy.Publisher(self.yolo_box_cloud_topic, PointCloud2, queue_size=1)
        self.current_door_cloud_pub = rospy.Publisher(self.current_door_cloud_topic, PointCloud2, queue_size=1)
        self.next_door_cloud_pub = rospy.Publisher(self.next_door_cloud_topic, PointCloud2, queue_size=1)
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
        if not self.valid_single_door_cloud(pts, depths):
            return None, pts

        seed = np.median(pts, axis=0)
        neighborhood = self.post_neighborhood_points(seed, cloud_world)
        if neighborhood.shape[0] >= self.min_points_in_box:
            pts = neighborhood

        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        keep = pts[dist < np.percentile(dist, 70)]
        if keep.shape[0] >= self.min_points_in_box:
            return np.median(keep, axis=0), keep
        return median, pts

    def post_neighborhood_points(self, seed, cloud_world):
        if cloud_world.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        half_xy = 0.5 * self.post_neighborhood_xy_m
        half_z = 0.5 * self.post_neighborhood_z_m
        delta = cloud_world - seed
        mask = (
            (np.abs(delta[:, 0]) <= half_xy) &
            (np.abs(delta[:, 1]) <= half_xy) &
            (np.abs(delta[:, 2]) <= half_z)
        )
        points = cloud_world[mask]
        if points.shape[0] < self.min_points_in_box:
            return points
        median = np.median(points, axis=0)
        dist = np.linalg.norm(points - median, axis=1)
        keep = dist < np.percentile(dist, 85)
        return points[keep]

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
        z_span = np.max(pts[:, 2]) - np.min(pts[:, 2])
        if z_span > self.single_door_height_m + self.single_door_height_tolerance_m:
            return False

        xy_span = np.ptp(pts[:, :2], axis=0)
        horizontal_span = float(np.max(xy_span))
        if horizontal_span > self.single_door_width_m + self.single_door_width_tolerance_m:
            return False

        depth_span = float(np.max(depths) - np.min(depths)) if depths.size else 0.0
        if depth_span > self.single_door_depth_m + self.single_door_depth_tolerance_m:
            return False
        return True

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
        for left_det, right_det, left_post, right_post in candidate_pairs:
            center = 0.5 * (left_post + right_post)
            width = np.linalg.norm((right_post - left_post)[:2])
            width_error = abs(width - self.door_width_m)
            if width_error > self.door_width_tolerance_m:
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
                "door_waypoint_detector: no valid 3D door pair from %d 2D-matched pairs",
                len(candidate_pairs))
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
        forward_world = self.drone_forward_world(odom)
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
                    track.left_post = 0.85 * track.left_post + 0.15 * refined
                    changed = True
            if track.right_post is not None:
                refined = self.refine_point_from_cloud(track.right_post, cloud_world)
                if refined is not None:
                    track.right_post = 0.85 * track.right_post + 0.15 * refined
                    changed = True
            if changed and track.left_post is not None and track.right_post is not None:
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
        return np.median(near, axis=0)

    def make_waypoints(self, track, odom):
        if track.center is None:
            return []
        return [self.waypoint_from_track(track)]

    def waypoint_from_track(self, track):
        waypoint = track.center.copy()
        waypoint[2] += self.waypoint_z_offset_m
        return waypoint

    def control_waypoints(self, current_track, next_track):
        waypoints = []
        if current_track is not None and current_track.center is not None:
            waypoints.append(self.waypoint_from_track(current_track))
        if next_track is not None and next_track.center is not None:
            next_waypoint = self.waypoint_from_track(next_track)
            if not waypoints or np.linalg.norm(next_waypoint - waypoints[-1]) > 0.5:
                waypoints.append(next_waypoint)
        return waypoints

    def publish_path(self, waypoints, stamp):
        path = Path()
        path.header.frame_id = "world"
        path.header.stamp = stamp
        for point in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = make_point(point)
            pose.pose.orientation.w = 1.0
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
