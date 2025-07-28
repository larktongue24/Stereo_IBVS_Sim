#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation

class StereoVisionProcessor:
    def __init__(self):
        rospy.init_node('stereo_vision_processor_node')
        rospy.loginfo("Stereo Vision Processor Node Started")

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.prev_gray_left = None
        self.p0_left = None  
        self.is_tracking_left = False

        self.prev_gray_right = None
        self.p0_right = None 
        self.is_tracking_right = False

        self.left_cam_info_received = False
        self.right_cam_info_received = False
        self.left_camera_matrix = None
        self.right_camera_matrix = None

        self.tcp_pixel_left = None
        self.tcp_pixel_right = None

        self.T_wrist3_tcp = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.25],
            [0, 0, 1, 0.1],
            [0, 0, 0, 1]
        ])
        rospy.loginfo(f"Virtual TCP defined relative to wrist_3_link.")

        self.T_wrist3_camL = self.create_transform_matrix(pos=[-0.1, 0.105, 0], euler_rad=[1.5708, 0, -0.2618])
        self.T_wrist3_camR = self.create_transform_matrix(pos=[0.1, 0.105, 0], euler_rad=[1.5708, 0, 0.2618])

        self.T_camL_wrist3 = np.linalg.inv(self.T_wrist3_camL)
        self.T_camR_wrist3 = np.linalg.inv(self.T_wrist3_camR)
        rospy.loginfo("Camera-to-EndEffector transforms calculated.")

        left_image_sub = message_filters.Subscriber('/mujoco_server/cameras/left_camera/rgb/image_raw', Image)
        right_image_sub = message_filters.Subscriber('/mujoco_server/cameras/right_camera/rgb/image_raw', Image)
        
        self.left_info_sub = rospy.Subscriber('/mujoco_server/cameras/left_camera/rgb/camera_info', CameraInfo, self.left_cam_info_callback)
        self.right_info_sub = rospy.Subscriber('/mujoco_server/cameras/right_camera/rgb/camera_info', CameraInfo, self.right_cam_info_callback)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([left_image_sub, right_image_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.corners_pub = rospy.Publisher('/stereo_aruco_corner_pixels', Float32MultiArray, queue_size=1)
        self.tcp_pixels_pub = rospy.Publisher('/stereo_virtual_tcp_pixels', Float32MultiArray, queue_size=1, latch=True)

        rospy.loginfo("Waiting for camera info...")

    def create_transform_matrix(self, pos, euler_rad):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', euler_rad).as_matrix()
        T[:3, 3] = pos
        return T

    def project_point_to_pixel(self, T_cam_point, K):
        P_cam = T_cam_point[:3, 3]
        if P_cam[2] <= 0: return None
        p_proj = K @ P_cam
        return (int(p_proj[0] / p_proj[2]), int(p_proj[1] / p_proj[2]))

    def left_cam_info_callback(self, msg):
        if not self.left_cam_info_received:
            self.left_camera_matrix = np.array(msg.K).reshape(3, 3)
            self.left_cam_info_received = True
            rospy.loginfo("Left camera intrinsics received.")
            self.calculate_tcp_projection()
            self.left_info_sub.unregister()

    def right_cam_info_callback(self, msg):
        if not self.right_cam_info_received:
            self.right_camera_matrix = np.array(msg.K).reshape(3, 3)
            self.right_cam_info_received = True
            rospy.loginfo("Right camera intrinsics received.")
            self.calculate_tcp_projection()
            self.right_info_sub.unregister()
            
    def calculate_tcp_projection(self):
        if self.left_cam_info_received and self.right_cam_info_received:
            rospy.loginfo("Both camera infos received. Calculating TCP projection...")
            T_camL_tcp = self.T_camL_wrist3 @ self.T_wrist3_tcp
            self.tcp_pixel_left = self.project_point_to_pixel(T_camL_tcp, self.left_camera_matrix)
            T_camR_tcp = self.T_camR_wrist3 @ self.T_wrist3_tcp
            self.tcp_pixel_right = self.project_point_to_pixel(T_camR_tcp, self.right_camera_matrix)

            if self.tcp_pixel_left and self.tcp_pixel_right:
                 rospy.loginfo(f"Virtual TCP projected to: Left Cam {self.tcp_pixel_left}, Right Cam {self.tcp_pixel_right}")
                 tcp_msg = Float32MultiArray()
                 tcp_msg.data = [self.tcp_pixel_left[0], self.tcp_pixel_left[1], 
                                 self.tcp_pixel_right[0], self.tcp_pixel_right[1]]
                 self.tcp_pixels_pub.publish(tcp_msg)
            else:
                 rospy.logwarn("Failed to project TCP, it might be behind the camera.")

    def image_callback(self, left_img_msg, right_img_msg):
        if not (self.left_cam_info_received and self.right_cam_info_received and self.tcp_pixel_left):
            rospy.loginfo_once("Waiting for all camera info and TCP projection to be ready...")
            return

        try:
            left_cv_image = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            left_corner = self.track_or_detect_feature(left_cv_image, 'left')
            self.draw_markers(left_cv_image, left_corner, self.tcp_pixel_left)
            cv2.imshow("Left Camera View", left_cv_image)

            right_cv_image = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            right_corner = self.track_or_detect_feature(right_cv_image, 'right')
            self.draw_markers(right_cv_image, right_corner, self.tcp_pixel_right)
            cv2.imshow("Right Camera View", right_cv_image)

            if left_corner is not None and right_corner is not None:
                msg = Float32MultiArray()
                msg.data = [left_corner[0], left_corner[1], right_corner[0], right_corner[1]]
                self.corners_pub.publish(msg)

            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def track_or_detect_feature(self, image, side):
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        is_tracking = self.is_tracking_left if side == 'left' else self.is_tracking_right
        prev_gray = self.prev_gray_left if side == 'left' else self.prev_gray_right
        p0 = self.p0_left if side == 'left' else self.p0_right

        if is_tracking:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **self.lk_params)
            
            if st is not None and st[0][0] == 1:
                detected_corner = (int(p1[0][0][0]), int(p1[0][0][1]))
            else: 
                rospy.logwarn(f"Tracking lost on {side} camera. Re-detecting...")
                is_tracking = False
                detected_corner = self.detect_feature(frame_gray)
        else: 
            detected_corner = self.detect_feature(frame_gray)

        if detected_corner is not None:
            is_tracking = True
            p0 = np.array([[detected_corner]], dtype=np.float32)
        else:
            is_tracking = False
            p0 = None
        
        if side == 'left':
            self.is_tracking_left = is_tracking
            self.p0_left = p0
            self.prev_gray_left = frame_gray.copy()
        else:
            self.is_tracking_right = is_tracking
            self.p0_right = p0
            self.prev_gray_right = frame_gray.copy()
            
        return detected_corner

    def detect_feature(self, gray_image):
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            first_corner = corners[0][0][0]
            return (int(first_corner[0]), int(first_corner[1]))
        return None

    def draw_markers(self, image, corner_pixel, tcp_pixel):
        if corner_pixel:
            cv2.circle(image, corner_pixel, 8, (0, 255, 0), -1) 

        if tcp_pixel:
            cv2.drawMarker(image, tcp_pixel, (0, 0, 255),  
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

if __name__ == '__main__':
    try:
        processor = StereoVisionProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()