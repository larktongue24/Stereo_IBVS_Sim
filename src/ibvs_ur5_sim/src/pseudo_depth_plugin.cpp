// pseudo_depth_plugin.cpp
#include <ibvs_ur5_sim/pseudo_depth_plugin.h>
#include <pluginlib/class_list_macros.h>
#include <random>

namespace ibvs_ur5_sim
{
bool StereoPseudoDepthPlugin::load(const mjModel *m, mjData *d)
{
    m_ = m;
    d_ = d;

    std::string left_camera_name, right_camera_name, aruco_marker_name;
    if (!nh_.getParam("left_camera_name", left_camera_name)) {
        ROS_ERROR("StereoPseudoDepthPlugin: Missing required parameter 'left_camera_name'.");
        return false;
    }
    if (!nh_.getParam("right_camera_name", right_camera_name)) {
        ROS_ERROR("StereoPseudoDepthPlugin: Missing required parameter 'right_camera_name'.");
        return false;
    }
    if (!nh_.getParam("aruco_marker_name", aruco_marker_name)) {
        ROS_ERROR("StereoPseudoDepthPlugin: Missing required parameter 'aruco_marker_name'.");
        return false;
    }
    if (!nh_.getParam("aruco_marker_size", aruco_size_)) {
        ROS_ERROR("StereoPseudoDepthPlugin: Missing required parameter 'aruco_marker_size'.");
        return false;
    }

    left_cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, left_camera_name.c_str());
    right_cam_id_ = mj_name2id(m_, mjOBJ_CAMERA, right_camera_name.c_str());
    aruco_body_id_ = mj_name2id(m_, mjOBJ_BODY, aruco_marker_name.c_str());

    if (left_cam_id_ == -1 || right_cam_id_ == -1 || aruco_body_id_ == -1) {
        ROS_ERROR("StereoPseudoDepthPlugin: Could not find one of the cameras or the aruco marker body.");
        if (left_cam_id_ == -1) ROS_ERROR(" -> Could not find left_camera: %s", left_camera_name.c_str());
        if (right_cam_id_ == -1) ROS_ERROR(" -> Could not find right_camera: %s", right_camera_name.c_str());
        if (aruco_body_id_ == -1) ROS_ERROR(" -> Could not find aruco_marker: %s", aruco_marker_name.c_str());
        return false;
    }

    left_depth_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/left_aruco_corners_pseudo_depth", 1);
    right_depth_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/right_aruco_corners_pseudo_depth", 1);
    
    ROS_INFO("StereoPseudoDepthPlugin loaded successfully.");

    double half_size = aruco_size_ / 2.0;
    corner_local_pos_[0][0] = -half_size; corner_local_pos_[0][1] =  half_size; corner_local_pos_[0][2] = 0;
    corner_local_pos_[1][0] =  half_size; corner_local_pos_[1][1] =  half_size; corner_local_pos_[1][2] = 0;
    corner_local_pos_[2][0] =  half_size; corner_local_pos_[2][1] = -half_size; corner_local_pos_[2][2] = 0;
    corner_local_pos_[3][0] = -half_size; corner_local_pos_[3][1] = -half_size; corner_local_pos_[3][2] = 0;

    return true;
}

void StereoPseudoDepthPlugin::reset() {}

void StereoPseudoDepthPlugin::controlCallback(const mjModel *m, mjData *d)
{

    const double NOISE_STDDEV = 0.0000001; 
    static std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(0.0, NOISE_STDDEV);

    mjtNum* left_cam_world_pos = d->cam_xpos + left_cam_id_ * 3;
    mjtNum* right_cam_world_pos = d->cam_xpos + right_cam_id_ * 3;

    mjtNum* aruco_world_pos = d->xpos + aruco_body_id_ * 3;
    mjtNum* aruco_world_mat = d->xmat + aruco_body_id_ * 9;

    mjtNum corner_world_pos[3];
    mjtNum rotated_corner[3];
    mju_rotVecMat(rotated_corner, corner_local_pos_[0], aruco_world_mat);
    mju_add(corner_world_pos, rotated_corner, aruco_world_pos, 3); 

    mjtNum left_distance = mju_dist3(corner_world_pos, left_cam_world_pos);
    if (NOISE_STDDEV > 0.0) {
        left_distance += distribution(generator);
        if (left_distance < 0) left_distance = 0;
    }

    mjtNum right_distance = mju_dist3(corner_world_pos, right_cam_world_pos);
    if (NOISE_STDDEV > 0.0) {
        right_distance += distribution(generator);
        if (right_distance < 0) right_distance = 0;
    }

    std_msgs::Float32MultiArray left_depth_msg;
    left_depth_msg.data.push_back(static_cast<float>(left_distance));
    left_depth_pub_.publish(left_depth_msg);

    std_msgs::Float32MultiArray right_depth_msg;
    right_depth_msg.data.push_back(static_cast<float>(right_distance));
    right_depth_pub_.publish(right_depth_msg);

}
} // namespace ibvs_ur5_sim

PLUGINLIB_EXPORT_CLASS(ibvs_ur5_sim::StereoPseudoDepthPlugin, mujoco_ros::MujocoPlugin)