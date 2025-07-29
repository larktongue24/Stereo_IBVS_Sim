#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.linalg import pinv
import message_filters
import tf2_ros
from tf.transformations import quaternion_from_matrix, translation_from_matrix, concatenate_matrices, translation_matrix, quaternion_matrix, quaternion_about_axis
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse

class StereoIBVSController:
    def __init__(self):
        rospy.init_node('stereo_ibvs_control_node')
        rospy.loginfo("Stereo IBVS Controller Node Started")

        self.lambda_ = 4.0  
        self.dt_ = 0.01
        self.rate = rospy.Rate(1 / self.dt_)
        self.error_threshold_ = 0.6

        self.servoing_active = False
        self.is_ready_to_servo = False
        self.home_joint_positions = [0.073981, -1.7, -0.6, -2.2, 1.473974, 0.2]

        self.left_cam_matrix = None
        self.right_cam_matrix = None
        self.f_l, self.cx_l, self.cy_l = None, None, None
        self.f_r, self.cx_r, self.cy_r = None, None, None
        self.R_rl = None  

        self.s_des_ = None 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.planning_group = "manipulator"
        self.base_frame = "base_link"
        self.tool_frame = "tool0"
        self.master_camera_frame = "left_camera_frame" 
        self.initial_tool_pose_matrix_ = None 

        self.joint_traj_client_ = actionlib.SimpleActionClient(
            '/scaled_pos_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for trajectory action server...")
        self.joint_traj_client_.wait_for_server()
        rospy.loginfo("Action server found.")
        
        ik_service_name = "/compute_ik"
        rospy.loginfo(f"Waiting for IK service: {ik_service_name}...")
        rospy.wait_for_service(ik_service_name)
        self.compute_ik_client = rospy.ServiceProxy(ik_service_name, GetPositionIK)
        rospy.loginfo("IK service found.")
        
        self.error_pub = rospy.Publisher('/ibvs_pixel_error', Float32, queue_size=10)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1)
        self.current_joint_positions = []
        self.joint_names = []

        left_depth_sub = message_filters.Subscriber('/left_aruco_corners_pseudo_depth', Float32MultiArray)
        right_depth_sub = message_filters.Subscriber('/right_aruco_corners_pseudo_depth', Float32MultiArray)
        corners_sub = message_filters.Subscriber('/stereo_aruco_corner_pixels', Float32MultiArray)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [corners_sub, left_depth_sub, right_depth_sub], 
            queue_size=10, slop=0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.ibvs_callback)

        self.left_info_sub = rospy.Subscriber("/mujoco_server/cameras/left_camera/rgb/camera_info", CameraInfo, self.left_camera_info_callback)
        self.right_info_sub = rospy.Subscriber("/mujoco_server/cameras/right_camera/rgb/camera_info", CameraInfo, self.right_camera_info_callback)
        
        rospy.Subscriber("/stereo_virtual_tcp_pixels", Float32MultiArray, self.desired_features_callback)

        self.start_service = rospy.Service('/ibvs/start_servoing', Trigger, self.handle_start_servoing)

        self.R_rl = self.get_stereo_rotation_matrix()
        rospy.loginfo(f"--- R_rl Rotation Matrix ---\n{np.round(self.R_rl, 3)}")
        if self.R_rl is None:
            rospy.logerr("Could not get stereo rotation matrix. Shutting down.")
            rospy.signal_shutdown("TF Error")

        rospy.loginfo("Stereo IBVS Controller initialized. Waiting for topics...")


    def joint_state_callback(self, msg):
        if not self.joint_names:
            self.joint_names = list(msg.name)
        
        positions = []
        for name in self.joint_names:
            try:
                index = msg.name.index(name)
                positions.append(msg.position[index])
            except ValueError:
                pass
        self.current_joint_positions = positions

    def left_camera_info_callback(self, msg):
        if self.left_cam_matrix is None:
            self.left_cam_matrix = np.array(msg.K).reshape(3, 3)
            self.f_l = self.left_cam_matrix[0, 0] 
            self.cx_l, self.cy_l = self.left_cam_matrix[0, 2], self.left_cam_matrix[1, 2]
            rospy.loginfo(f"Left camera intrinsics received.")
            self.left_info_sub.unregister()

    def right_camera_info_callback(self, msg):
        if self.right_cam_matrix is None:
            self.right_cam_matrix = np.array(msg.K).reshape(3, 3)
            self.f_r = self.right_cam_matrix[0, 0]
            self.cx_r, self.cy_r = self.right_cam_matrix[0, 2], self.right_cam_matrix[1, 2]
            rospy.loginfo(f"Right camera intrinsics received.")
            self.right_info_sub.unregister()

    def desired_features_callback(self, msg):
        if self.s_des_ is None:
            self.s_des_ = np.array(msg.data)
            rospy.loginfo(f"Desired stereo features (s_des) received: {self.s_des_}")

    def handle_start_servoing(self, req):
        rospy.loginfo("Start servoing command received!")

        try:
            rospy.loginfo("Capturing initial tool orientation as the desired orientation...")
            transform_stamped = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rospy.Time(0), rospy.Duration(1.0))
            self.initial_tool_pose_matrix_ = self.transform_to_matrix(transform_stamped)
            rospy.loginfo("Initial orientation captured successfully.")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to capture initial tool pose: {e}")
            return TriggerResponse(success=False, message=f"Failed to get initial pose: {e}")

        self.servoing_active = True
        return TriggerResponse(success=True, message="Visual servoing has been activated.")

    def ibvs_callback(self, corners_msg, left_depth_msg, right_depth_msg):
        if not self.servoing_active:
            rospy.loginfo_throttle(5.0, "IBVS is standing by, waiting for start command...")
            return
        
        if not self.is_ready_to_servo:
            if not self.current_joint_positions:
                rospy.logwarn_throttle(1.0, "Waiting for initial joint states to check home position...")
                return

            joint_error = np.linalg.norm(np.array(self.current_joint_positions) - np.array(self.home_joint_positions))
            
            if joint_error < 1.5:
                rospy.loginfo("Robot has reached home position. Starting IBVS control loop.")
                self.is_ready_to_servo = True 
            else:
                rospy.loginfo_throttle(1.0, f"Waiting for robot to reach home position. Current joint error: {joint_error:.3f}")
                return
        
        if self.left_cam_matrix is None or self.right_cam_matrix is None or self.s_des_ is None or self.R_rl is None:
            rospy.logwarn_throttle(1.0, "IBVS: Waiting for all camera info, desired features, and stereo transform...")
            return

        s_cur = np.array(corners_msg.data) 
        # Z_l = left_depth_msg.data[0]
        # Z_r = right_depth_msg.data[0]

        Z_l = 0.302655
        Z_r = 0.302655

        # if Z_l < 0.01 or Z_r < 0.01:
        #     rospy.logwarn("Invalid depth value detected, skipping step.")
        #     return

        error = s_cur - self.s_des_ 
        
        avg_pixel_error = np.mean([np.linalg.norm(error[0:2]), np.linalg.norm(error[2:4])])
        self.error_pub.publish(avg_pixel_error)
        rospy.loginfo(f"Average pixel error: {avg_pixel_error:.2f}")

        if avg_pixel_error < self.error_threshold_:
            rospy.loginfo_once(f"Target reached! Final average pixel error: {avg_pixel_error:.2f}")
            self.servoing_active = False 

            self.verify_convergence_in_tool_frame()

            return
        
        J_stereo = self.compute_stereo_image_jacobian(s_cur, Z_l, Z_r)

        k = 0.002
        # J_pseudo_inv = pinv(J_stereo)
        # J_pseudo_inv = J_stereo.T @ np.linalg.inv(J_stereo @ J_stereo.T + k * np.identity(4))

        JtJ = J_stereo.T @ J_stereo
        I = np.identity(3) 
        J_pseudo_inv = np.linalg.inv(JtJ + k * I) @ J_stereo.T

        if avg_pixel_error > 25:
            self.lambda_ = 5
        else:
            if avg_pixel_error > 10:
                self.lambda_ = 10.0
            else:
                if avg_pixel_error > 5:
                    self.lambda_ = 20.0
                else:
                    if avg_pixel_error > 2:
                        self.lambda_ = 30.0
                    else:
                        if avg_pixel_error > 1:
                            self.lambda_ = 40.0
                        else:
                            self.lambda_ = 50.0

        v_c_linear = -self.lambda_ * (J_pseudo_inv @ error)

        rospy.loginfo(f"Current velocity: {v_c_linear[0]:.2f}, {v_c_linear[1]:.2f}, {v_c_linear[2]:.2f}")

        v_cam = np.zeros(6)
        v_cam[0:3] = v_c_linear

        try:
            trans_base_to_cam = self.tf_buffer.lookup_transform(self.base_frame, self.master_camera_frame, rospy.Time(0), rospy.Duration(0.1))
            trans_tool_to_cam = self.tf_buffer.lookup_transform(self.tool_frame, self.master_camera_frame, rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}"); return

        T_base_to_cam = self.transform_to_matrix(trans_base_to_cam)
        T_tool_to_cam = self.transform_to_matrix(trans_tool_to_cam)
        T_cam_to_tool = np.linalg.inv(T_tool_to_cam)

        linear_vel, angular_vel = v_cam[0:3], v_cam[3:6]
        delta_translation = linear_vel * self.dt_
        
        T_inc_rotation = np.identity(4)
        T_inc_translation = translation_matrix(delta_translation)
        T_inc_cam = concatenate_matrices(T_inc_translation, T_inc_rotation)

        T_target_cam_in_base = T_base_to_cam @ T_inc_cam
        T_target_tool_in_base = T_target_cam_in_base @ T_cam_to_tool

        if self.initial_tool_pose_matrix_ is not None:
            T_target_tool_in_base[:3, :3] = self.initial_tool_pose_matrix_[:3, :3]
        else:
            rospy.logwarn_throttle(1.0, "Initial orientation not set, pose may drift!")

        self.solve_and_execute_ik(T_target_tool_in_base)

    def compute_stereo_image_jacobian(self, s_cur, Z_l, Z_r):
        u_l, v_l, u_r, v_r = s_cur

        u_l_norm = u_l - self.cx_l
        v_l_norm = v_l - self.cy_l

        J_l = np.array([
            [-self.f_l / Z_l, 0, u_l_norm / Z_l],
            [0, -self.f_l / Z_l, v_l_norm / Z_l]
        ])

        J_r = np.zeros((2, 3))
        r1 = self.R_rl[0, :] 
        r2 = self.R_rl[1, :] 
        r3 = self.R_rl[2, :] 
        
        u_r_norm = u_r - self.cx_r
        v_r_norm = v_r - self.cy_r

        J_r[0, :] = -( (self.f_r / Z_r) * r1 - (u_r_norm / Z_r) * r3 )
        J_r[1, :] = -( (self.f_r / Z_r) * r2 - (v_r_norm / Z_r) * r3 )

        return np.vstack((J_l, J_r))

    def get_stereo_rotation_matrix(self):
        rospy.loginfo("Attempting to get transform from left_camera_frame to right_camera_frame (R_rl)...")
        while not rospy.is_shutdown():
            try:
                self.tf_buffer.can_transform('right_camera_frame', 'left_camera_frame', rospy.Time(0.1), rospy.Duration(2.0))

                transform_stamped = self.tf_buffer.lookup_transform(
                    'right_camera_frame',  # Target Frame
                    'left_camera_frame',   # Source Frame
                    rospy.Time(0.1)
                )
                
                rospy.loginfo("Successfully received transform R_rl.")
                rotation_q = transform_stamped.transform.rotation
                q_list = [rotation_q.x, rotation_q.y, rotation_q.z, rotation_q.w]

                return quaternion_matrix(q_list)[:3, :3]

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(2.0, f"Waiting for transform R_rl: {e}")
                rospy.sleep(1.0)
        return None
    
    def solve_and_execute_ik(self, target_pose_matrix):
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.planning_group
        
        robot_state = RobotState()
        robot_state.joint_state.name = self.joint_names
        robot_state.joint_state.position = self.current_joint_positions
        req.ik_request.robot_state = robot_state

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pos_target = translation_from_matrix(target_pose_matrix)
        rot_target = quaternion_from_matrix(target_pose_matrix)
        pose_stamped.pose.position.x = pos_target[0]
        pose_stamped.pose.position.y = pos_target[1]
        pose_stamped.pose.position.z = pos_target[2]
        pose_stamped.pose.orientation.x = rot_target[0]
        pose_stamped.pose.orientation.y = rot_target[1]
        pose_stamped.pose.orientation.z = rot_target[2]
        pose_stamped.pose.orientation.w = rot_target[3]
        
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.timeout = rospy.Duration(0.05)
        req.ik_request.avoid_collisions = False

        try:
            response = self.compute_ik_client(req)
            if response.error_code.val == response.error_code.SUCCESS:
                goal = FollowJointTrajectoryGoal()
                goal.trajectory.joint_names = self.filter_joint_names_for_controller(response.solution.joint_state.name)
                point = JointTrajectoryPoint()
                point.positions = self.filter_joint_positions_for_controller(response.solution.joint_state.name, list(response.solution.joint_state.position))
                point.time_from_start = rospy.Duration(self.dt_ * 0.9)
                goal.trajectory.points.append(point)
                self.joint_traj_client_.send_goal(goal)
            else:
                rospy.logwarn_throttle(1.0, f"IK FAILED with error code: {response.error_code.val}")
        except rospy.ServiceException as e:
            rospy.logerr(f"IK service call failed: {e}")

    def transform_to_matrix(self, transform_stamped):
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        return concatenate_matrices(translation_matrix([t.x, t.y, t.z]), 
                                      quaternion_matrix([r.x, r.y, r.z, r.w]))

    def filter_joint_names_for_controller(self, solution_joint_names):
        controller_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        return [name for name in solution_joint_names if name in controller_joints]
        
    def filter_joint_positions_for_controller(self, solution_joint_names, solution_positions):
        controller_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        filtered_positions = []
        for name in controller_joints:
            try:
                index = solution_joint_names.index(name)
                filtered_positions.append(solution_positions[index])
            except ValueError:
                rospy.logerr(f"Joint '{name}' not found in IK solution!")
        return filtered_positions
    
    def verify_convergence_in_tool_frame(self):
        """
        Verifies the final 3D position by calculating the Aruco corner's position
        in the tool frame and comparing it to the virtual TCP's known position.
        """
        rospy.loginfo("--- Starting Final Pose Verification in Tool Frame ---")
        try:
            # 1. Define the world coordinates of the tracked Aruco corner
            P_world_corner = np.array([0.45, 0.05, 0.2])
            rospy.loginfo(f"Static world position of tracked Aruco corner: {P_world_corner}")

            # Define the rotation from world to base (180 deg around Z)
            # This matrix rotates a point from the world frame to the base_link frame.
            R_base_world = np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]])

            # First, transform the corner from MuJoCo world to the robot's base_link frame
            P_base_corner = R_base_world @ P_world_corner

            # 2. Get the transform from base_link to tool0
            trans_tool = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rospy.Time(0), rospy.Duration(1.0))
            T_base_tool0 = self.transform_to_matrix(trans_tool)
            T_tool0_base = np.linalg.inv(T_base_tool0)

            # 3. Transform the corner's position (now in base_link) into the tool frame
            P_base_corner_homogeneous = np.append(P_base_corner, 1) # Use the corrected point
            P_tool0_corner_homogeneous = T_tool0_base @ P_base_corner_homogeneous
            P_tool0_corner = P_tool0_corner_homogeneous[:3]
            rospy.loginfo(f"Calculated Aruco corner position in tool frame: {P_tool0_corner}")

            # 4. Compare with the known virtual TCP position
            P_tool0_tcp = np.array([0, 0.05, 0.3])
            P_tool0_tcp_homogeneous = np.append(P_tool0_tcp, 1)
            P_base_tcp_homogeneous = T_base_tool0 @ P_tool0_tcp_homogeneous
            P_base_tcp = P_base_tcp_homogeneous[:3]
            rospy.loginfo(f"Target virtual TCP position in base frame:      {P_base_tcp}")

            physical_error_m = np.linalg.norm(P_tool0_corner - P_tool0_tcp)
            
            rospy.loginfo("-------------------------------------------")
            rospy.loginfo(f"Final 3D Physical Error: {physical_error_m * 1000:.2f} mm")
            rospy.loginfo("-------------------------------------------")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Error during verification: {e}")

if __name__ == '__main__':
    try:
        controller = StereoIBVSController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass