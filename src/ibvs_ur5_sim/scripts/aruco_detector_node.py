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
import tf2_ros
from tf.transformations import concatenate_matrices, translation_matrix, quaternion_matrix

class StereoVisionProcessor:
    def __init__(self):
        rospy.init_node('stereo_vision_processor_node')
        rospy.loginfo("Stereo Vision Processor Node Started")

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.left_cam_info_received = False
        self.right_cam_info_received = False
        self.left_camera_matrix = None
        self.right_camera_matrix = None

        self.tcp_pixel_left = None
        self.tcp_pixel_right = None


        self.T_wrist3_tcp = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.2], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # self.T_wrist3_camL = self.create_transform_matrix(pos=[-0.02, 0.105, 0], euler_rad=[1.5708, 0, -0.2618])
        # self.T_wrist3_camR = self.create_transform_matrix(pos=[0.02, 0.105, 0], euler_rad=[1.5708, 0, 0.2618])
        self.T_wrist3_camL = None
        self.T_wrist3_camR = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        wrist_frame = "wrist_3_link"
        left_cam_frame = "left_camera_frame"
        right_cam_frame = "right_camera_frame"
        rospy.loginfo("Attempting to get camera extrinsics from TF tree...")
        try:

            self.tf_buffer.can_transform(left_cam_frame, wrist_frame, rospy.Time(0), rospy.Duration(5.0))
            trans_L = self.tf_buffer.lookup_transform(left_cam_frame, wrist_frame, rospy.Time(0), rospy.Duration(1.0))
            self.T_camL_wrist3 = self.transform_to_matrix(trans_L)
            rospy.loginfo("Successfully received T_wrist3_camL from TF.")

            self.tf_buffer.can_transform(right_cam_frame, wrist_frame, rospy.Time(0), rospy.Duration(5.0))
            trans_R = self.tf_buffer.lookup_transform(right_cam_frame, wrist_frame, rospy.Time(0), rospy.Duration(1.0))
            self.T_camR_wrist3 = self.transform_to_matrix(trans_R)
            rospy.loginfo("Successfully received T_wrist3_camR from TF.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"CRITICAL TF ERROR during initialization: {e}")
            rospy.signal_shutdown("Could not get essential camera transforms from TF.")

        rospy.loginfo("Camera-to-EndEffector transforms calculated.")

        left_image_sub = message_filters.Subscriber('/mujoco_server/cameras/left_camera/rgb/image_raw', Image)
        right_image_sub = message_filters.Subscriber('/mujoco_server/cameras/right_camera/rgb/image_raw', Image)
        
        rospy.Subscriber('/mujoco_server/cameras/left_camera/rgb/camera_info', CameraInfo, self.left_cam_info_callback)
        rospy.Subscriber('/mujoco_server/cameras/right_camera/rgb/camera_info', CameraInfo, self.right_cam_info_callback)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([left_image_sub, right_image_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.corners_pub = rospy.Publisher('/stereo_aruco_corner_pixels', Float32MultiArray, queue_size=1)
        self.tcp_pixels_pub = rospy.Publisher('/stereo_virtual_tcp_pixels', Float32MultiArray, queue_size=1, latch=True)

        rospy.loginfo("Waiting for camera info...")

    def transform_to_matrix(self, transform_stamped):
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        return concatenate_matrices(translation_matrix([t.x, t.y, t.z]), 
                                      quaternion_matrix([r.x, r.y, r.z, r.w]))
    
    def create_transform_matrix(self, pos, euler_rad):

        T = np.eye(4)
        # T[:3, :3] = Rotation.from_euler('xyz', euler_rad).as_matrix()
        T[:3, :3] = Rotation.from_euler('zxy', [euler_rad[2], euler_rad[0], euler_rad[1]]).as_matrix()
        T[:3, 3] = pos
        return T


    def project_point_to_pixel(self, T_cam_point, K):

        P_cam = T_cam_point[:3, 3]
        

        if P_cam[2] <= 0:
            return None
        
        p_proj = K @ P_cam
        pixel_x = p_proj[0] / p_proj[2]
        pixel_y = p_proj[1] / p_proj[2]
        
        return (int(pixel_x), int(pixel_y))

    # def project_point_to_pixel(self, T_cam_point, K):
    #     """
    #     Projects a 3D point in the camera frame to a 2D pixel coordinate.
    #     This version correctly handles the Z-axis-backward camera convention.
    #     """
    #     P_cam = T_cam_point[:3, 3]  # P_cam = [X, Y, Z]

    #     # Check if the point is in front of the camera (where Z is negative)
    #     # Use a small epsilon for floating point safety
    #     if P_cam[2] <= 1e-6:
    #         return None
        
    #     # Extract intrinsic parameters from the K matrix
    #     fx = K[0, 0]
    #     fy = K[1, 1]
    #     cx = K[0, 2]
    #     cy = K[1, 2]

    #     # Correct projection formula for a Z-backward camera system
    #     # The depth 'd' is positive, so we use -P_cam[2]
    #     d = P_cam[2]
    #     pixel_x = fx * (P_cam[0] / d) + cx
    #     pixel_y = fy * (P_cam[1] / d) + cy
        
    #     return (int(pixel_x), int(pixel_y))
    

    def left_cam_info_callback(self, msg):
        if not self.left_cam_info_received:
            self.left_camera_matrix = np.array(msg.K).reshape(3, 3)
            self.left_cam_info_received = True
            rospy.loginfo("Left camera intrinsics received.")
            self.calculate_tcp_projection()

    def right_cam_info_callback(self, msg):
        if not self.right_cam_info_received:
            self.right_camera_matrix = np.array(msg.K).reshape(3, 3)
            self.right_cam_info_received = True
            rospy.loginfo("Right camera intrinsics received.")
            self.calculate_tcp_projection()
            
    def calculate_tcp_projection(self):

        if self.left_cam_info_received and self.right_cam_info_received:

            rospy.loginfo("Both camera infos received. Calculating TCP projection...")

            rospy.loginfo("--- DEBUG INFO ---")
            rospy.loginfo(f"Left K Matrix:\n{self.left_camera_matrix}")
            rospy.loginfo(f"Right K Matrix:\n{self.right_camera_matrix}")
            
            T_camL_tcp = self.T_camL_wrist3 @ self.T_wrist3_tcp
            P_camL = T_camL_tcp[:3, 3]
            rospy.loginfo(f"TCP in Left Cam Frame (X, Y, Z): {P_camL}")
            self.tcp_pixel_left = self.project_point_to_pixel(T_camL_tcp, self.left_camera_matrix)

            T_camR_tcp = self.T_camR_wrist3 @ self.T_wrist3_tcp
            P_camR = T_camR_tcp[:3, 3]
            rospy.loginfo(f"TCP in Right Cam Frame (X, Y, Z): {P_camR}")
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
            left_cv_image_raw = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            right_cv_image_raw = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")

            left_corner = self.process_single_image(left_cv_image_raw)
            right_corner = self.process_single_image(right_cv_image_raw)

            left_display_image = left_cv_image_raw.copy()
            right_display_image = right_cv_image_raw.copy()

            self.draw_visualizations(left_display_image, left_corner, self.tcp_pixel_left)
            self.draw_visualizations(right_display_image, right_corner, self.tcp_pixel_right)

            cv2.imshow("Left Camera View", left_display_image)
            cv2.imshow("Right Camera View", right_display_image)

            if left_corner is not None and right_corner is not None:
                msg = Float32MultiArray()
                msg.data = [left_corner[0], left_corner[1], right_corner[0], right_corner[1]]
                self.corners_pub.publish(msg)

            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def process_single_image(self, image):
        detected_corner = None
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            first_corner = corners[0][0][0]
            detected_corner = (int(first_corner[0]), int(first_corner[1]))
            
        return detected_corner
    
    def draw_visualizations(self, image, detected_corner, tcp_pixel):
        if detected_corner:
            cv2.circle(image, detected_corner, 8, (0, 255, 0), -1)

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