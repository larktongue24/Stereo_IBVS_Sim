#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped

br = None
last_stamp = None

def pose_callback(msg):

    global br, last_stamp

    if msg.header.stamp == last_stamp:
        return

    last_stamp = msg.header.stamp

    if br is None:
        return

    t = TransformStamped()

    t.header.stamp = msg.header.stamp
    t.header.frame_id = "base"
    t.child_frame_id = "wrist_3_link_gt" 

    t.transform.translation = msg.pose.position
    t.transform.rotation = msg.pose.orientation

    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('ground_truth_tf_broadcaster')

    br = tf2_ros.TransformBroadcaster()
    
    rospy.loginfo("Ground Truth TF Broadcaster started. Publishing unique 'wrist_3_link_gt'.")
    
    rospy.Subscriber('/mujoco_ground_truth/wrist_3_pose', PoseStamped, pose_callback)
    
    rospy.spin()