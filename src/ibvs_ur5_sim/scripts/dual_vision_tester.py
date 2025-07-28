#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class DualVisionTester:
    def __init__(self):
        rospy.init_node('dual_vision_tester_node', anonymous=True)
        rospy.loginfo("Dual Vision Tester Node Started")

        self.bridge = CvBridge()

        left_rgb_topic = "/mujoco_server/cameras/left_camera/rgb/image_raw"
        right_rgb_topic = "/mujoco_server/cameras/right_camera/rgb/image_raw"

        rospy.loginfo("Subscribing to topics:")
        rospy.loginfo(f"  - Left RGB: {left_rgb_topic}")
        rospy.loginfo(f"  - Right RGB: {right_rgb_topic}")

        left_sub = message_filters.Subscriber(left_rgb_topic, Image)
        right_sub = message_filters.Subscriber(right_rgb_topic, Image)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], 
            queue_size=10, 
            slop=0.1  
        )

        self.ts.registerCallback(self.image_callback)

    def image_callback(self, left_msg, right_msg):
        rospy.loginfo_once("Successfully received synchronized stereo images! (This message will only appear once)")

        try:
            cv_left_image = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
            cv_right_image = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")

            cv2.imshow("Left Camera View", cv_left_image)
            cv2.imshow("Right Camera View", cv_right_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        tester = DualVisionTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()