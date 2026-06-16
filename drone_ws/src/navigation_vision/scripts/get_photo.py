#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from pathlib import Path

class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        self.bridge = CvBridge()
        self.latest_image = None
        self.display = rospy.get_param('~display', True)
        self.autosave = rospy.get_param('~autosave', False)
        self.save_interval = rospy.get_param('~save_interval', 1.0)
        self.max_images = rospy.get_param('~max_images', 0)
        self.last_save_time = rospy.Time(0)
        nav_vision_dir = Path(__file__).resolve().parents[1]
        default_save_dir = nav_vision_dir / 'yolov8' / 'images' / 'train'
        self.save_dir = rospy.get_param('~save_dir', str(default_save_dir))
        self.topic = rospy.get_param('~topic', '/airsim_node/drone_1/front_right/Scene')
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Get the starting index based on existing files
        self.current_index = self.get_starting_index()
        
        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber(
            self.topic,
            Image,
            self.image_callback
        )
        
        print(f"Image saver initialized. Topic: {self.topic}")
        print(f"Saving images to: {self.save_dir}")
        if self.display:
            print("Press SPACE to save images, ESC to quit.")
        if self.autosave:
            print(f"Autosave enabled: every {self.save_interval:.2f}s, max_images={self.max_images or 'unlimited'}")
        
    def get_starting_index(self):
        # Get list of existing image files
        existing_files = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        if not existing_files:
            return 0
        
        # Extract indices from filenames and find the maximum
        indices = []
        for filename in existing_files:
            try:
                index = int(filename.split('.')[0])
                indices.append(index)
            except ValueError:
                continue
        
        return max(indices) + 1 if indices else 0
    
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def save_image(self):
        if self.latest_image is None:
            return

        filename = os.path.join(self.save_dir, f"{self.current_index:04d}.jpg")
        cv2.imwrite(filename, self.latest_image)
        print(f"Saved image {filename}")
        self.current_index += 1
            
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                if self.autosave:
                    now = rospy.Time.now()
                    if (now - self.last_save_time).to_sec() >= self.save_interval:
                        self.save_image()
                        self.last_save_time = now
                        if self.max_images > 0 and self.current_index >= self.max_images:
                            break

                if self.display:
                    cv2.imshow('Camera Feed', self.latest_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Space bar
                        self.save_image()
                    elif key == 27:  # ESC
                        break
            rate.sleep()
        
        if self.display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        saver = ImageSaver()
        saver.run()
    except rospy.ROSInterruptException:
        pass
