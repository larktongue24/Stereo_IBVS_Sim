// pseudo_depth_plugin.h
#pragma once

#include <mujoco_ros/plugin_utils.h>
#include <ros/ros.h>
#include <mujoco/mujoco.h>
#include <std_msgs/Float32MultiArray.h>

namespace ibvs_ur5_sim
{
class StereoPseudoDepthPlugin : public mujoco_ros::MujocoPlugin
{
public:
    ~StereoPseudoDepthPlugin() override = default;

    // Overload entry point
    bool load(const mjModel *m, mjData *d) override;
    // Called on reset
    void reset() override;
    // Called on every control step
    void controlCallback(const mjModel *m, mjData *d) override;

private:
    ros::NodeHandle nh_;
    ros::Publisher left_depth_pub_;
    ros::Publisher right_depth_pub_;

    const mjModel *m_;
    mjData *d_;

    int left_cam_id_;
    int right_cam_id_;
    int aruco_body_id_;
    double aruco_size_; 

    mjtNum corner_local_pos_[4][3];
};
} // namespace ibvs_ur5_sim