#!/usr/bin/env python3

import os
import threading
from pathlib import Path

os.environ["GDAL_DATA"] = ""
os.environ["PROJ_LIB"] = ""

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


class YoloDoorDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.latest_image = None
        self.latest_stamp = rospy.Time(0)
        self.running = True

        self.model_path = rospy.get_param("~model_path", "/drone_ws/src/navigation_vision/yolov8/rmua_v2.pt")
        self.image_topic = rospy.get_param("~image_topic", "/airsim_node/drone_1/front_left/Scene")
        self.detections_topic = rospy.get_param("~detections_topic", "/door_waypoint_detector/yolo_detections")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/door_waypoint_detector/yolo_debug_image")
        self.confidence_threshold = float(rospy.get_param("~confidence_threshold", 0.6))
        self.nms_iou_threshold = float(rospy.get_param("~nms_iou_threshold", 0.45))
        self.model_input_size = int(rospy.get_param("~model_input_size", 640))
        self.publish_debug_image = bool(rospy.get_param("~publish_debug_image", False))
        self.imshow_raw_detection = bool(rospy.get_param("~imshow_raw_detection", True))
        self.imshow_window_name = rospy.get_param("~imshow_window_name", "yolo_raw_detection")

        self.model = YOLO(self.model_path)
        try:
            self.model.fuse()
        except Exception as exc:
            rospy.logwarn("yolo_door_detector: ultralytics fuse skipped: %s", exc)
        rospy.loginfo(
            "yolo_door_detector: Ultralytics model=%s imgsz=%d detections=%s",
            self.model_path,
            self.model_input_size,
            self.detections_topic)

        self.det_pub = rospy.Publisher(self.detections_topic, Float32MultiArray, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.worker = threading.Thread(target=self.worker_loop)
        self.worker.daemon = True
        self.worker.start()

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "yolo_door_detector: cv_bridge failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_stamp = msg.header.stamp
        self.frame_event.set()

    def worker_loop(self):
        while not rospy.is_shutdown() and self.running:
            self.frame_event.wait(0.2)
            self.frame_event.clear()
            with self.lock:
                if self.latest_image is None:
                    continue
                image = self.latest_image.copy()
                stamp = self.latest_stamp
                self.latest_image = None
            detections, raw_detections = self.detect(image)
            msg = Float32MultiArray()
            data = [stamp.to_sec(), float(len(detections))]
            for det in detections:
                data.extend([det["class_id"], det["conf"], *det["bbox"].tolist()])
            msg.data = data
            self.det_pub.publish(msg)
            if self.imshow_raw_detection:
                self.show_raw_detection(image, raw_detections)
            if self.publish_debug_image:
                self.publish_debug(image, raw_detections, stamp)

    def detect(self, image):
        results = self.model.predict(
            image,
            imgsz=self.model_input_size,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False)
        detections = []
        raw_detections = []
        if not results or results[0].boxes is None:
            return detections, raw_detections
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        for bbox, conf, cls in zip(xyxy, confs, classes):
            raw_detections.append({
                "class_id": int(cls),
                "conf": float(conf),
                "bbox": np.array(bbox, dtype=np.float64),
            })
            if cls in (0, 1):
                x1, y1, x2, y2 = bbox
                if (x2 - x1) * (y2 - y1) >= 50.0:
                    detections.append({
                        "class_id": int(cls),
                        "conf": float(conf),
                        "bbox": np.array([x1, y1, x2, y2], dtype=np.float64),
                    })
        return detections, raw_detections

    def show_raw_detection(self, image, raw_detections):
        vis = image.copy()
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            color = (255, 180, 30) if det["class_id"] == 0 else (30, 120, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class_id"]}:{det["conf"]:.2f}'
            cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow(self.imshow_window_name, vis)
        cv2.waitKey(1)

    def publish_debug(self, image, detections, stamp):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            color = (255, 180, 30) if det["class_id"] == 0 else (30, 120, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class_id"]}:{det["conf"]:.2f}'
            cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out = self.bridge.cv2_to_imgmsg(image, "bgr8")
        out.header.stamp = stamp
        out.header.frame_id = "front_left"
        self.debug_pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("yolo_door_detector")
    YoloDoorDetector()
    rospy.spin()
