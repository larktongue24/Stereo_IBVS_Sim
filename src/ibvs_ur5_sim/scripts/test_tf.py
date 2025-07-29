#!/usr/bin/env python3
import rospy
import tf2_ros

rospy.init_node('tf_wrist_checker')
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

rate = rospy.Rate(1.0)
while not rospy.is_shutdown():
    try:
        trans = tf_buffer.lookup_transform('base', 'wrist_3_link', rospy.Time(0))
        
        t = trans.transform.translation
        r = trans.transform.rotation
        
        print("--- TF Model Pose (world -> wrist_3_link) ---")
        print(f"  Pos: [ {t.x: .4f}, {t.y: .4f}, {t.z: .4f} ]")

        print(f"  Quat: [ {r.x: .4f}, {r.y: .4f}, {r.z: .4f}, {r.w: .4f} ]")
        print("-" * 20)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn_throttle(2.0, f"Waiting for TF transform: {e}")
    
    rate.sleep()