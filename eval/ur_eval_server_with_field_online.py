#!/usr/bin/env python3
"""
UR Real Robot Adaptive Hybrid Control Script
- Gradient Field always online
- If gradient > threshold: use Field control
- If gradient < threshold: use Policy control
"""

import sys
import os
import rospy
import numpy as np
import cv2
import zmq
import time
import torch
import torch.nn.functional as F
from collections import deque
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from termcolor import cprint

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import robotiq_gripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from model.eqm_with_gradient_vis import GradientFieldNetwork, PureEqMModel

# =====================================================================
# Config
# =====================================================================
ROBOT_HOST = "192.168.56.101"

# Server addresses
SEG_ADDR = "tcp://192.168.56.55:5556"
ACTION_BIND = "tcp://192.168.56.60:5557"
POSE_ADDR = "tcp://192.168.56.55:5557"

# Gradient field model
MODEL_PATH = "/home/ani/RSS2026/GradientField/log/banana_pick.pth"

# Control parameters
OBS_STEPS = 1
IMG_STEPS = 1
CONTROL_HZ = 10

# Gradient field parameters
GRAD_NORM_THRESHOLD = 0.015
GRAD_STEP_SIZE = 0.1

# Camera-TCP extrinsic
CAMERA_TCP_EXTRINSIC = np.array([-0.1841869894889, 0.0017635353786, -0.1552642302730, 
                                 -0.1884937273, 0.1684104182, -0.6796188446, 0.6886408875])

# Motion parameters
MOVE_SPEED = 0.05
MOVE_ACCELERATION = 0.15

# Vision update frequency for field control
VISION_UPDATE_EVERY_N_STEPS = 5

# =====================================================================
# Utility Functions
# =====================================================================
def rotvec_to_quat_xyzw(rx, ry, rz):
    rotvec = np.array([rx, ry, rz])
    return R.from_rotvec(rotvec).as_quat()


def quat_xyzw_to_rotvec(qx, qy, qz, qw):
    quat_xyzw = np.array([qx, qy, qz, qw])
    return R.from_quat(quat_xyzw).as_rotvec()


def ensure_quaternion_continuity(quat, last_quat):
    if last_quat is None:
        return quat
    dot_product = np.dot(quat, last_quat)
    if dot_product < 0:
        quat = -quat
    return quat


def grad_to_delta_T(grad, step_size):
    delta_pos = grad[:3] * step_size
    delta_rotvec = grad[3:] * step_size
    
    T_delta = np.eye(4)
    if np.linalg.norm(delta_rotvec) > 1e-8:
        T_delta[:3, :3] = R.from_rotvec(delta_rotvec).as_matrix()
    T_delta[:3, 3] = delta_pos
    
    return T_delta


# =====================================================================
# Model Loading
# =====================================================================
def load_gradient_field_model(model_path, device):
    cprint(f"Loading gradient field model from {model_path}...", "cyan")
    
    network = GradientFieldNetwork(
        input_dim=7,
        output_dim=6,
        hidden_dims=[256, 512, 512, 512, 256],
    )
    
    dummy_traj_pos = torch.zeros(1, 3).to(device)
    dummy_traj_quat = torch.tensor([[0, 0, 0, 1]]).float().to(device)
    
    model = PureEqMModel(
        network=network,
        traj_pos=dummy_traj_pos,
        traj_quat=dummy_traj_quat,
        pos_weight=1.0,
        rot_weight=1.0,
        magnitude_type='distance',
        inference_step_size=GRAD_STEP_SIZE,
        max_inference_steps=100,
        grad_norm_threshold=GRAD_NORM_THRESHOLD,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.network.eval()
    
    cprint("Model loaded successfully!", "green")
    return model


# =====================================================================
# Adaptive Hybrid Control Node
# =====================================================================
class AdaptiveHybridControlNode:
    def __init__(self):
        rospy.init_node("ur_adaptive_hybrid_control", anonymous=True)
        
        self.bridge = CvBridge()
        
        rospy.loginfo("Connecting to robot...")
        self.rtde_c = RTDEControlInterface(ROBOT_HOST)
        self.rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        
        rospy.loginfo("Connecting to gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(ROBOT_HOST, 63352)
        self.gripper.activate()
        
        # ZMQ sockets
        self.ctx = zmq.Context.instance()
        
        self.seg_sock = self.ctx.socket(zmq.REQ)
        self.seg_sock.connect(SEG_ADDR)
        rospy.loginfo(f"Connected to SegServer at {SEG_ADDR}")

        self.pose_sock = self.ctx.socket(zmq.REQ)
        self.pose_sock.connect(POSE_ADDR)
        rospy.loginfo(f"Connected to PoseServer at {POSE_ADDR}")
        
        self.action_sock = self.ctx.socket(zmq.REP)
        self.action_sock.bind(ACTION_BIND)
        rospy.loginfo(f"Waiting for Server on {ACTION_BIND}")
        
        # Gradient field model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {device}")
        self.device = device
        self.grad_model = load_gradient_field_model(MODEL_PATH, device)
        
        # Camera-TCP transform
        cam_pos = CAMERA_TCP_EXTRINSIC[:3]
        cam_quat = CAMERA_TCP_EXTRINSIC[3:]
        self.R_tcp_to_cam = R.from_quat(cam_quat).as_matrix()
        self.t_tcp_to_cam = cam_pos
        
        # Observation buffers
        self.state_buf = deque(maxlen=OBS_STEPS)
        self.img_buf = deque(maxlen=IMG_STEPS)
        self.depth_buf = deque(maxlen=IMG_STEPS)
        
        # Action queue for policy
        self.action_queue = deque()
        
        # State tracking
        self.last_quat = None
        self.step_count = 0
        self.current_obj_pose = None
        self.last_tcp_pose = None
        
        # Control mode tracking
        self.current_mode = "field"  # "field" or "policy"
        
        # ROS subscribers
        wrist_color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        wrist_depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [wrist_color_sub, wrist_depth_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=True
        )
        self.ts.registerCallback(self.observation_callback)
        
        rospy.loginfo("Adaptive hybrid control node initialized")
    
    
    def observation_callback(self, wrist_color_msg, wrist_depth_msg):
        wrist_img = self.bridge.imgmsg_to_cv2(wrist_color_msg, "bgr8")
        wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        wrist_depth = self.bridge.imgmsg_to_cv2(wrist_depth_msg)
        
        eef_pose = self.rtde_r.getActualTCPPose()
        quat_xyzw = rotvec_to_quat_xyzw(eef_pose[3], eef_pose[4], eef_pose[5])
        quat_xyzw = ensure_quaternion_continuity(quat_xyzw, self.last_quat)
        self.last_quat = quat_xyzw.copy()
        
        gripper_width = self.gripper.get_current_position()
        state = np.array([gripper_width])
        
        self.state_buf.append(state)
        self.img_buf.append(wrist_img)
        self.depth_buf.append(wrist_depth)
    
    
    def get_observation(self):
        if len(self.state_buf) < OBS_STEPS or len(self.img_buf) < IMG_STEPS or len(self.depth_buf) < IMG_STEPS:
            return None
        
        obs = {
            "state": np.stack(self.state_buf, axis=0),
            "image": np.stack(self.img_buf, axis=0),
            "depth": np.stack(self.depth_buf, axis=0),
        }
        return obs
    
    
    def request_observation_with_obj_pose(self, force_request=False):
        obs = self.get_observation()
        if obs is None:
            return None
        
        need_obj_pose = force_request or (self.current_obj_pose is None)
        
        obs["require_obj_pose"] = need_obj_pose
        obs["require_action"] = False

        self.pose_sock.send_pyobj(obs)

        if need_obj_pose:
            result = self.pose_sock.recv_pyobj()
            
            if "obj_pose" in result:
                self.current_obj_pose = result["obj_pose"]
                obs["obj_pose"] = self.current_obj_pose
                
                tcp_pose = self.rtde_r.getActualTCPPose()
                self.last_tcp_pose = np.array(tcp_pose)
            else:
                rospy.logwarn("obj_pose not in result!")
                return None
        else:
            self.pose_sock.recv()
            
            if self.current_obj_pose is None:
                rospy.logwarn("No obj_pose available yet!")
                return None
            
            obs["obj_pose"] = self.current_obj_pose
        
        return obs
    

    def update_obj_pose_from_tcp_motion(self):
        if self.current_obj_pose is None or self.last_tcp_pose is None:
            return
        
        current_tcp_pose = self.rtde_r.getActualTCPPose()
        current_tcp_pose = np.array(current_tcp_pose)
        
        tcp_delta_pos = current_tcp_pose[:3] - self.last_tcp_pose[:3]
        
        R_tcp_old = R.from_rotvec(self.last_tcp_pose[3:]).as_matrix()
        R_tcp_new = R.from_rotvec(current_tcp_pose[3:]).as_matrix()
        R_tcp_delta = R_tcp_new @ R_tcp_old.T
        
        R_cam_to_tcp = self.R_tcp_to_cam.T
        
        obj_delta_pos_cam = -R_cam_to_tcp @ tcp_delta_pos
        
        R_obj_delta_cam = R_cam_to_tcp @ R_tcp_delta @ self.R_tcp_to_cam
        
        obj_pos = self.current_obj_pose[:3]
        obj_quat = self.current_obj_pose[3:]
        
        obj_pos_new = obj_pos + obj_delta_pos_cam
        
        R_obj_old = R.from_quat(obj_quat).as_matrix()
        R_obj_new = R_obj_delta_cam @ R_obj_old
        obj_quat_new = R.from_matrix(R_obj_new).as_quat()
        
        self.current_obj_pose = np.concatenate([obj_pos_new, obj_quat_new])
        self.last_tcp_pose = current_tcp_pose


    def request_actions(self, obs):
        obs["require_action"] = True
        obs["require_obj_pose"] = False
        
        self.seg_sock.send_pyobj(obs)
        self.seg_sock.recv()
        
        result = self.action_sock.recv_pyobj()
        self.action_sock.send(b"ok")
        return result
    
    
    def send_obs_only(self, obs):
        """Send observation to seg server without requesting action"""
        obs["require_action"] = False
        obs["require_obj_pose"] = False
        
        self.seg_sock.send_pyobj(obs)
        self.seg_sock.recv()


    # =====================================================================
    # Gradient Field Control
    # =====================================================================
    
    def predict_gradient(self, obj_pose_camera):
        pose_tensor = torch.from_numpy(obj_pose_camera).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            grad = self.grad_model.network(pose_tensor)
            grad_norm = grad.norm(dim=1).item()
        
        grad_np = grad.squeeze(0).cpu().numpy()
        
        return grad_np, grad_norm
    
    
    def compute_tcp_delta_from_gradient(self, grad):
        T_grad_cam = grad_to_delta_T(grad, GRAD_STEP_SIZE)
        
        T_grad_cam_inv = np.linalg.inv(T_grad_cam)
        R_grad_cam = T_grad_cam_inv[:3, :3]
        t_grad_cam = T_grad_cam_inv[:3, 3]
        
        tcp_pose = self.rtde_r.getActualTCPPose()
        tcp_rotvec = np.array(tcp_pose[3:])
        R_base_to_tcp = R.from_rotvec(tcp_rotvec).as_matrix()
        
        R_delta_tcp = self.R_tcp_to_cam @ R_grad_cam @ self.R_tcp_to_cam.T
        t_delta_tcp = R_base_to_tcp @ self.R_tcp_to_cam @ t_grad_cam
        
        tcp_delta_rotvec = R.from_matrix(R_delta_tcp).as_rotvec()
        tcp_delta_pos = t_delta_tcp
        
        return tcp_delta_pos, tcp_delta_rotvec
    
    
    def apply_field_tcp_delta(self, delta_pos, delta_rotvec):
        current_pose = self.rtde_r.getActualTCPPose()
        current_pos = np.array(current_pose[:3])
        current_rotvec = np.array(current_pose[3:])
        
        target_pos = current_pos + delta_pos
        current_rot = R.from_rotvec(current_rotvec)
        delta_rot = R.from_rotvec(delta_rotvec)
        target_rot = current_rot * delta_rot
        target_rotvec = target_rot.as_rotvec()
        
        target_pose = np.concatenate([target_pos, target_rotvec])
        
        try:
            self.rtde_c.moveL(target_pose.tolist(), 
                            speed=MOVE_SPEED, 
                            acceleration=MOVE_ACCELERATION)
            
            self.update_obj_pose_from_tcp_motion()
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to execute field motion: {e}")
            return False
    
    
    # =====================================================================
    # Policy Control
    # =====================================================================
    
    def apply_policy_action(self, action):
        current_pose = self.rtde_r.getActualTCPPose()
        current_pos = np.array(current_pose[:3])
        current_quat = rotvec_to_quat_xyzw(current_pose[3], current_pose[4], current_pose[5])
        
        target_pos = current_pos + action[:3]
        target_quat = current_quat + action[3:7]
        target_rot = R.from_quat(target_quat)
        target_rotvec = target_rot.as_rotvec()
        
        target_pose = np.concatenate([target_pos, target_rotvec])
        
        try:
            self.rtde_c.moveL(target_pose.tolist(), 
                            speed=MOVE_SPEED, 
                            acceleration=MOVE_ACCELERATION)
            
            gripper_width = int(np.clip(np.round(action[7]), 0, 255))
            self.gripper.move(gripper_width, 155, 155)
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to execute policy action: {e}")
            return False
    
    
    # =====================================================================
    # Main Adaptive Control Loop
    # =====================================================================
    
    def run(self):
        """Main adaptive control loop - switches between field and policy based on gradient"""
        cprint("\n" + "="*60, "magenta", attrs=["bold"])
        cprint("ADAPTIVE HYBRID CONTROL", "magenta", attrs=["bold"])
        cprint(f"  Gradient threshold: {GRAD_NORM_THRESHOLD}", "magenta")
        cprint(f"  grad > threshold -> Field control", "magenta")
        cprint(f"  grad < threshold -> Policy control", "magenta")
        cprint("="*60 + "\n", "magenta", attrs=["bold"])
        
        rate = rospy.Rate(CONTROL_HZ)
        
        while not rospy.is_shutdown():
            self.step_count += 1
            
            # Get observation with obj_pose (for gradient computation)
            force_vision_update = (self.step_count % VISION_UPDATE_EVERY_N_STEPS == 0)
            obs = self.request_observation_with_obj_pose(force_request=force_vision_update)
            
            if obs is None or "obj_pose" not in obs:
                rospy.logwarn("No observation available, waiting...")
                rate.sleep()
                continue
            
            obj_pose_camera = obs["obj_pose"]
            
            # Predict gradient
            grad, grad_norm = self.predict_gradient(obj_pose_camera)
            
            # Decide control mode based on gradient
            if grad_norm > GRAD_NORM_THRESHOLD:
                # Use Field control
                new_mode = "field"
                
                if self.current_mode != new_mode:
                    cprint(f"\n[Step {self.step_count}] Switching to FIELD control", "green", attrs=["bold"])
                    self.action_queue.clear()
                
                self.current_mode = new_mode
                
                # Compute and apply field control
                tcp_delta_pos, tcp_delta_rotvec = self.compute_tcp_delta_from_gradient(grad)
                delta_euler = R.from_rotvec(tcp_delta_rotvec).as_euler("ZYX", degrees=True)
                
                cprint(f"[Step {self.step_count}] FIELD | grad_norm={grad_norm:.6f} | pos_delta={np.linalg.norm(tcp_delta_pos):.6f}", "green")
                
                # Send obs to seg server (keep it updated)
                self.send_obs_only(obs)
                
                success = self.apply_field_tcp_delta(tcp_delta_pos, tcp_delta_rotvec)
                
                if not success:
                    cprint("Field motion failed", "red")
                    break
                    
            else:
                # Use Policy control
                new_mode = "policy"
                
                if self.current_mode != new_mode:
                    cprint(f"\n[Step {self.step_count}] Switching to POLICY control", "blue", attrs=["bold"])
                    self.action_queue.clear()
                
                self.current_mode = new_mode
                
                # Request new actions if queue is empty
                if len(self.action_queue) == 0:
                    try:
                        result = self.request_actions(obs)
                        action_seq = result['action']
                        
                        rospy.loginfo(f"[Step {self.step_count}] Received {len(action_seq)} actions from policy")
                        
                        for action in action_seq:
                            self.action_queue.append(action)
                    
                    except Exception as e:
                        rospy.logerr(f"Failed to get actions from policy: {e}")
                        import traceback
                        traceback.print_exc()
                        rate.sleep()
                        continue
                else:
                    # Send obs to seg server even if not requesting action
                    self.send_obs_only(obs)
                
                # Execute action from queue
                if len(self.action_queue) > 0:
                    action = self.action_queue.popleft()
                    
                    cprint(f"[Step {self.step_count}] POLICY | grad_norm={grad_norm:.6f} | queue={len(self.action_queue)}", "blue")
                    
                    success = self.apply_policy_action(action)
                    
                    if not success:
                        cprint("Policy action failed", "red")
                        break
            
            rate.sleep()
        
        rospy.loginfo("Adaptive control loop ended")
    
    
    def shutdown(self):
        rospy.loginfo("Shutting down...")
        self.rtde_c.stopScript()
        self.gripper.disconnect()
        self.seg_sock.close()
        self.pose_sock.close()
        self.action_sock.close()
        self.ctx.term()


# =====================================================================
# Main
# =====================================================================
def main():
    try:
        node = AdaptiveHybridControlNode()
        
        rospy.loginfo("Waiting for observations...")
        while len(node.state_buf) < OBS_STEPS and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Observations ready!")
        
        # Run adaptive control loop
        node.run()
        
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.shutdown()


if __name__ == "__main__":
    main()