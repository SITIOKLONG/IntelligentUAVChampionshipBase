#!/usr/bin/env python3

import os
import sys
import rospy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from door_waypoint_detector import DoorFusionTracker


if __name__ == "__main__":
    rospy.init_node("door_fusion_tracker")
    DoorFusionTracker()
    rospy.spin()
