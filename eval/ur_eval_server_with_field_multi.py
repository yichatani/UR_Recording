#!/usr/bin/env python3
"""
UR Real Robot Hybrid Control Script
- Phase 1: Gradient Field Coarse Positioning
- Phase 2: Policy Fine Control
- Observations from remote server via ZMQ
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
SEG_ADDR = "tcp://192.168.56.55:5556"      # Send obs to seg server
ACTION_BIND = "tcp://192.168.56.60:5557"   # Receive action/obj_pose from server
POSE_ADDR = "tcp://192.168.56.55:5557"

# Gradient field model
# MODEL_PATH = "/home/ani/RSS2026/GradientField/log/banana_pick.pth"
# MODEL_PATH = "/home/ani/RSS2026/GradientField/log/open_canvas.pth"

MODEL_PATH = ["/home/ani/RSS2026/GradientField/log/open_canvas.pth",
                "/home/ani/RSS2026/GradientField/log/banana_pick.pth"]

END_OPEN_GRIPPER = [True, False]  # Whether to open gripper at end of each stage

# Control parameters
OBS_STEPS = 1
IMG_STEPS = 1
CONTROL_HZ = 10

# Gradient field parameters
GRAD_NORM_THRESHOLD = 0.005
GRAD_STEP_SIZE = 0.1
MAX_GRAD_ITERATIONS = 150

# Policy parameters
STOP_THRESHOLD = 0.5
CUMULATIVE_STOP_COUNT = 2

# Termination detection parameters
TERMINATION_WINDOW_SIZE = 5
TERMINATION_POS_DELTA_THRESHOLD = 0.0005  # 0.5mm - based on log showing ~0.0001-0.0003 at convergence
TERMINATION_QUAT_DELTA_THRESHOLD = 0.0002  # based on log showing ~0.00002-0.00008 at convergence
TERMINATION_GRIPPER_DELTA_THRESHOLD = 2.0  # gripper units - based on log showing stable ~137
TERMINATION_STOP_THRESHOLD = 0.99

# Camera-TCP extrinsic
# Format: [x, y, z, qx, qy, qz, qw]
CAMERA_TCP_EXTRINSIC = np.array([-0.1841869894889, 0.0017635353786, -0.1552642302730, 
                                 -0.1884937273, 0.1684104182, -0.6796188446, 0.6886408875])

# Motion parameters
MOVE_SPEED = 0.15
MOVE_ACCELERATION = 0.3

INITIAL_JOINT = np.array([-0.4882410208331507, -1.0156464141658326, -1.6417837142944336, 
                              -1.2842187744430085, 1.7657989263534546, 1.3109745979309082])  # Example initial pose [x, y, z, rx, ry, rz]

# =====================================================================
# Utility Functions
# =====================================================================
def rotvec_to_quat_xyzw(rx, ry, rz):
    """Convert rotation vector to quaternion [qx, qy, qz, qw]"""
    rotvec = np.array([rx, ry, rz])
    return R.from_rotvec(rotvec).as_quat()


def quat_xyzw_to_rotvec(qx, qy, qz, qw):
    """Convert quaternion [qx, qy, qz, qw] to rotation vector"""
    quat_xyzw = np.array([qx, qy, qz, qw])
    return R.from_quat(quat_xyzw).as_rotvec()


def ensure_quaternion_continuity(quat, last_quat):
    """Ensure quaternion continuity"""
    if last_quat is None:
        return quat
    dot_product = np.dot(quat, last_quat)
    if dot_product < 0:
        quat = -quat
    return quat


def grad_to_delta_T(grad, step_size):
    """Convert gradient to SE(3) delta transformation"""
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
    """Load trained EqM model"""
    cprint(f"Loading gradient field model from {model_path}...", "cyan")
    
    # Create network
    network = GradientFieldNetwork(
        input_dim=7,
        output_dim=6,
        hidden_dims=[256, 512, 512, 512, 256],
    )
    
    # Create model (需要dummy trajectory数据来初始化)
    # 因为你的模型在inference时不需要trajectory，所以这里用placeholder
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
        max_inference_steps=MAX_GRAD_ITERATIONS,
        grad_norm_threshold=GRAD_NORM_THRESHOLD,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.network.eval()
    
    cprint("Model loaded successfully!", "green")
    return model


# =====================================================================
# Hybrid Control Node
# =====================================================================
class HybridControlNode:
    def __init__(self):
        # ROS node
        rospy.init_node("ur_hybrid_control", anonymous=True)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Robot interfaces
        rospy.loginfo("Connecting to robot...")
        self.rtde_c = RTDEControlInterface(ROBOT_HOST)
        self.rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        
        # Initial robot pose
        self.rtde_c.moveJ(INITIAL_JOINT, speed=0.5, acceleration=0.25)

        # Gripper
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
        
        # Gradient field models (one per stage)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {device}")
        self.device = device
        self.grad_models = [load_gradient_field_model(path, device) for path in MODEL_PATH]
        
        # Current stage
        self.current_stage = 0
        
        # Camera-TCP transform
        cam_pos = CAMERA_TCP_EXTRINSIC[:3]
        cam_quat = CAMERA_TCP_EXTRINSIC[3:]
        self.R_tcp_to_cam = R.from_quat(cam_quat).as_matrix()
        self.t_tcp_to_cam = cam_pos
        
        # Observation buffers
        self.state_buf = deque(maxlen=OBS_STEPS)
        self.img_buf = deque(maxlen=IMG_STEPS)
        self.depth_buf = deque(maxlen=IMG_STEPS)
        
        # Action queue
        self.action_queue = deque()
        
        # State tracking
        self.last_quat = None
        self.step_count = 0
        self.total_stop_count = 0
        
        # Termination detection window
        self.termination_window = deque(maxlen=TERMINATION_WINDOW_SIZE)
        
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
        
        self.current_obj_pose = None
        self.last_tcp_pose = None

        rospy.loginfo("Hybrid control node initialized")
    
    
    def observation_callback(self, wrist_color_msg, wrist_depth_msg):
        """Process incoming observations"""
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
        """Get stacked observation"""
        if len(self.state_buf) < OBS_STEPS or len(self.img_buf) < IMG_STEPS or len(self.depth_buf) < IMG_STEPS:
            return None
        
        obs = {
            "state": np.stack(self.state_buf, axis=0),
            "image": np.stack(self.img_buf, axis=0),
            "depth": np.stack(self.depth_buf, axis=0),
            "stage": self.current_stage,
        }
        return obs
    
    
    def request_observation_with_obj_pose(self, force_request=False):
        """Request observation with obj_pose from remote server
        
        Args:
            force_request: if True, always request new obj_pose from vision
                          if False, only request if we don't have one yet
        
        Returns:
            obs: dict with "obj_pose" key
        """
        obs = self.get_observation()
        if obs is None:
            return None
        
        # Decide whether to request new obj_pose from vision
        need_obj_pose = force_request or (self.current_obj_pose is None)
        
        obs["require_obj_pose"] = need_obj_pose
        obs["require_action"] = False

        print("Requesting observation with obj_pose, need_obj_pose =", need_obj_pose)

        # Send to pose server
        self.pose_sock.send_pyobj(obs)
        
        print("Sent observation to pose server")

        if need_obj_pose:
            # Receive result with obj_pose from vision
            result = self.pose_sock.recv_pyobj()
            
            if "obj_pose" in result:
                self.current_obj_pose = result["obj_pose"]
                obs["obj_pose"] = self.current_obj_pose
                
                # Record current TCP pose for future delta calculation
                tcp_pose = self.rtde_r.getActualTCPPose()
                self.last_tcp_pose = np.array(tcp_pose)
                
                rospy.loginfo("Updated obj_pose from vision")
            else:
                rospy.logwarn("obj_pose not in result!")
                return None
        else:
            self.pose_sock.recv()  # Acknowledge without receiving new obj_pose

            # Use internally tracked obj_pose (updated by motion)
            if self.current_obj_pose is None:
                rospy.logwarn("No obj_pose available yet!")
                return None
            
            obs["obj_pose"] = self.current_obj_pose
            rospy.logdebug("Using tracked obj_pose")
        
        return obs
    

    def update_obj_pose_from_tcp_motion(self):
        """Update internally tracked obj_pose based on TCP motion
        
        This propagates the object pose forward using the TCP delta.
        Call this after each TCP motion to keep obj_pose synchronized.
        """
        if self.current_obj_pose is None or self.last_tcp_pose is None:
            return
        
        # Get current TCP pose
        current_tcp_pose = self.rtde_r.getActualTCPPose()
        current_tcp_pose = np.array(current_tcp_pose)
        
        # Compute TCP delta in base frame
        tcp_delta_pos = current_tcp_pose[:3] - self.last_tcp_pose[:3]
        tcp_delta_rotvec = current_tcp_pose[3:] - self.last_tcp_pose[3:]
        
        # Convert to rotation matrices
        R_tcp_old = R.from_rotvec(self.last_tcp_pose[3:]).as_matrix()
        R_tcp_new = R.from_rotvec(current_tcp_pose[3:]).as_matrix()
        R_tcp_delta = R_tcp_new @ R_tcp_old.T
        
        # Transform TCP delta from base frame to camera frame
        # Camera moves opposite to TCP in camera frame
        R_cam_to_tcp = self.R_tcp_to_cam.T
        
        # Position delta in camera frame
        obj_delta_pos_cam = -R_cam_to_tcp @ tcp_delta_pos
        
        # Rotation delta in camera frame
        R_obj_delta_cam = R_cam_to_tcp @ R_tcp_delta @ self.R_tcp_to_cam
        
        # Apply delta to current obj pose
        obj_pos = self.current_obj_pose[:3]
        obj_quat = self.current_obj_pose[3:]
        
        # Update position
        obj_pos_new = obj_pos + obj_delta_pos_cam
        
        # Update rotation
        R_obj_old = R.from_quat(obj_quat).as_matrix()
        R_obj_new = R_obj_delta_cam @ R_obj_old
        obj_quat_new = R.from_matrix(R_obj_new).as_quat()
        
        # Update internal state
        self.current_obj_pose = np.concatenate([obj_pos_new, obj_quat_new])
        self.last_tcp_pose = current_tcp_pose
        
        rospy.logdebug(f"Updated obj_pose via motion: pos_delta={np.linalg.norm(obj_delta_pos_cam):.6f}")
    


    def request_actions(self, obs, require_action=False):
        """Request actions from policy server
        
        Args:
            obs: observation dict
            require_action: whether to request action prediction from policy
        
        Returns:
            result: dict with 'action' key if require_action=True, else None
        """
        # Add flag to indicate whether we need action
        obs["require_action"] = require_action
        obs["require_obj_pose"] = False  # Policy phase doesn't need obj_pose
        
        # Send obs to seg server (always happens)
        self.seg_sock.send_pyobj(obs)
        self.seg_sock.recv()
        
        # Only receive action result if we requested it
        if require_action:
            result = self.action_sock.recv_pyobj()
            self.action_sock.send(b"ok")
            return result
        else:
            return None

    # =====================================================================
    # Phase 1: Gradient Field
    # =====================================================================
    
    def predict_gradient(self, obj_pose_camera):
        """Predict gradient from object pose
        
        Args:
            obj_pose_camera: (7,) [x, y, z, qx, qy, qz, qw]
        
        Returns:
            grad: (6,) gradient
            grad_norm: scalar
            converged: bool
        """
        pose_tensor = torch.from_numpy(obj_pose_camera).float().unsqueeze(0).to(self.device)
        
        # Use the model for current stage
        grad_model = self.grad_models[self.current_stage]
        
        with torch.no_grad():
            grad = grad_model.network(pose_tensor)  # (1, 6)
            grad_norm = grad.norm(dim=1).item()
        
        grad_np = grad.squeeze(0).cpu().numpy()
        converged = grad_norm < GRAD_NORM_THRESHOLD
        
        return grad_np, grad_norm, converged
    
    
    def compute_tcp_delta_from_gradient(self, grad):
        """Compute TCP delta from gradient in camera frame"""
        # Get gradient transformation (object motion in camera frame)
        T_grad_cam = grad_to_delta_T(grad, GRAD_STEP_SIZE)
        
        # Invert to get camera motion relative to object
        T_grad_cam_inv = np.linalg.inv(T_grad_cam)
        R_grad_cam = T_grad_cam_inv[:3, :3]
        t_grad_cam = T_grad_cam_inv[:3, 3]
        
        # Get current TCP pose
        tcp_pose = self.rtde_r.getActualTCPPose()
        tcp_pos = np.array(tcp_pose[:3])
        tcp_rotvec = np.array(tcp_pose[3:])
        R_base_to_tcp = R.from_rotvec(tcp_rotvec).as_matrix()
        
        # Transform to TCP frame, then to base frame
        R_delta_tcp = self.R_tcp_to_cam @ R_grad_cam @ self.R_tcp_to_cam.T
        t_delta_tcp = R_base_to_tcp @ self.R_tcp_to_cam @ t_grad_cam
        
        tcp_delta_rotvec = R.from_matrix(R_delta_tcp).as_rotvec()
        tcp_delta_pos = t_delta_tcp
        
        return tcp_delta_pos, tcp_delta_rotvec
    
    
    def apply_field_tcp_delta(self, delta_pos, delta_rotvec):
        """Apply TCP delta and update tracked obj_pose"""
        current_pose = self.rtde_r.getActualTCPPose()
        current_pos = np.array(current_pose[:3])
        current_rotvec = np.array(current_pose[3:])
        
        # Apply delta
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
            
            # Update obj_pose after motion
            self.update_obj_pose_from_tcp_motion()
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to execute motion: {e}")
            return False
    
    
    def run_gradient_field_phase(self, request_obj_pose_every_n_steps=5):
        """Phase 1: Gradient field coarse positioning
        
        Args:
            request_obj_pose_every_n_steps: Request new obj_pose from vision every N steps
                                            Set to 1 to request every step (vision-based)
                                            Set to >1 to rely on motion tracking between requests
        """
        cprint("\n" + "="*60, "green", attrs=["bold"])
        cprint(f"PHASE 1: Gradient Field Coarse Positioning (Stage {self.current_stage})", "green", attrs=["bold"])
        cprint(f"  Vision update frequency: every {request_obj_pose_every_n_steps} steps", "green")
        cprint("="*60 + "\n", "green", attrs=["bold"])
        
        # rate = rospy.Rate(CONTROL_HZ)
        
        for iteration in range(MAX_GRAD_ITERATIONS):
            # Decide whether to request new obj_pose from vision
            force_vision_update = (iteration % request_obj_pose_every_n_steps == 0)
            
            # Get observation with obj_pose
            obs = self.request_observation_with_obj_pose(force_request=force_vision_update)
            
            # print(f"{obs=}")
            # exit()

            if obs is None or "obj_pose" not in obs:
                rospy.logwarn("No observation available, waiting...")
                # rate.sleep()
                continue
            
            obj_pose_camera = obs["obj_pose"]
            
            # Predict gradient
            grad, grad_norm, converged = self.predict_gradient(obj_pose_camera)
            
            cprint(f"\n[Stage {self.current_stage} | Iteration {iteration + 1}/{MAX_GRAD_ITERATIONS}]", "cyan")
            print(f"  Object pose (cam): pos={obj_pose_camera[:3]}")
            print(f"  Gradient norm: {grad_norm:.6f}")
            print(f"  Source: {'Vision' if force_vision_update else 'Motion tracking'}")
            
            # Check convergence
            if converged:
                cprint(f"\nConverged at iteration {iteration + 1}!", "green", attrs=["bold"])
                cprint(f"  Final gradient norm: {grad_norm:.6f} < {GRAD_NORM_THRESHOLD}", "green")
                return True
            
            # Compute and apply TCP delta
            tcp_delta_pos, tcp_delta_rotvec = self.compute_tcp_delta_from_gradient(grad)
            delta_euler = R.from_rotvec(tcp_delta_rotvec).as_euler("ZYX", degrees=True)
            print(f"  TCP delta: pos_norm={np.linalg.norm(tcp_delta_pos):.6f}, euler_ZYX={delta_euler}")
            
            success = self.apply_field_tcp_delta(tcp_delta_pos, tcp_delta_rotvec)
            
            if not success:
                cprint("Motion failed", "red")
                return False
            
            # rate.sleep()
        
        cprint(f"\nReached max iterations ({MAX_GRAD_ITERATIONS})", "yellow")
        return False
    
    
    # =====================================================================
    # Phase 2: Policy Control
    # =====================================================================
    
    def check_termination_condition(self, action):
        """Check if policy should terminate based on sliding window of actions
        
        Args:
            action: (9,) [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, stop]
        
        Returns:
            should_terminate: bool
        """
        pos_delta_norm = np.linalg.norm(action[:3])
        quat_delta_norm = np.linalg.norm(action[3:7])
        gripper_value = action[7]
        stop_value = action[8]
        
        # Add current action metrics to window
        self.termination_window.append({
            'pos_delta_norm': pos_delta_norm,
            'quat_delta_norm': quat_delta_norm,
            'gripper': gripper_value,
            'stop': stop_value
        })
        
        # Need full window to check
        if len(self.termination_window) < TERMINATION_WINDOW_SIZE:
            return False
        
        # Check all conditions across window
        window_list = list(self.termination_window)
        
        # 1. All position deltas should be small
        all_pos_small = all(
            w['pos_delta_norm'] < TERMINATION_POS_DELTA_THRESHOLD 
            for w in window_list
        )
        
        # 2. All quaternion deltas should be small
        all_quat_small = all(
            w['quat_delta_norm'] < TERMINATION_QUAT_DELTA_THRESHOLD 
            for w in window_list
        )
        
        # 3. Gripper should be stable (small variation)
        gripper_values = [w['gripper'] for w in window_list]
        gripper_range = max(gripper_values) - min(gripper_values)
        gripper_stable = gripper_range < TERMINATION_GRIPPER_DELTA_THRESHOLD
        
        # 4. All stop values should be high
        all_stop_high = all(
            w['stop'] >= TERMINATION_STOP_THRESHOLD 
            for w in window_list
        )
        
        should_terminate = all_pos_small and all_quat_small and gripper_stable and all_stop_high
        
        if should_terminate:
            cprint(f"\n{'='*60}", "green", attrs=["bold"])
            cprint("TERMINATION CONDITION MET!", "green", attrs=["bold"])
            cprint(f"  Window size: {TERMINATION_WINDOW_SIZE} steps", "green")
            cprint(f"  Max pos_delta in window: {max(w['pos_delta_norm'] for w in window_list):.6f} < {TERMINATION_POS_DELTA_THRESHOLD}", "green")
            cprint(f"  Max quat_delta in window: {max(w['quat_delta_norm'] for w in window_list):.6f} < {TERMINATION_QUAT_DELTA_THRESHOLD}", "green")
            cprint(f"  Gripper range in window: {gripper_range:.2f} < {TERMINATION_GRIPPER_DELTA_THRESHOLD}", "green")
            cprint(f"  Min stop in window: {min(w['stop'] for w in window_list):.4f} >= {TERMINATION_STOP_THRESHOLD}", "green")
            cprint(f"{'='*60}\n", "green", attrs=["bold"])
        
        return should_terminate
    
    
    def apply_policy_action(self, action):
        """Apply policy action
        
        Args:
            action: (9,) [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, stop]
        """
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
            rospy.logerr(f"Failed to execute action: {e}")
            return False
    
    
    def run_policy_phase(self, mode="open_loop", actions_per_inference=10):
        """Phase 2: Policy fine control"""
        cprint("\n" + "="*60, "blue", attrs=["bold"])
        cprint(f"PHASE 2: Policy Fine Control ({mode}) (Stage {self.current_stage})", "blue", attrs=["bold"])
        cprint(f"  Termination window: {TERMINATION_WINDOW_SIZE} steps", "blue")
        cprint(f"  Termination thresholds:", "blue")
        cprint(f"    pos_delta < {TERMINATION_POS_DELTA_THRESHOLD}", "blue")
        cprint(f"    quat_delta < {TERMINATION_QUAT_DELTA_THRESHOLD}", "blue")
        cprint(f"    gripper_range < {TERMINATION_GRIPPER_DELTA_THRESHOLD}", "blue")
        cprint(f"    stop >= {TERMINATION_STOP_THRESHOLD}", "blue")
        cprint("="*60 + "\n", "blue", attrs=["bold"])
        
        # Clear termination window at start
        self.termination_window.clear()
        
        # rate = rospy.Rate(CONTROL_HZ)
        actions_executed_since_last_inference = 0
        
        while not rospy.is_shutdown():
            self.step_count += 1
            
            obs = self.get_observation()
            if obs is None:
                # rate.sleep()
                continue
            
            # Check if need new actions
            need_new_inference = False
            if mode == "open_loop":
                need_new_inference = (len(self.action_queue) == 0)
            elif mode == "close_loop":
                print(f"{len(self.action_queue)=}, {actions_executed_since_last_inference=}, {actions_per_inference=}")
                need_new_inference = (
                    len(self.action_queue) == 0 or
                    actions_executed_since_last_inference >= actions_per_inference
                )
                
            
            print(f"\n[Stage {self.current_stage} | Step {self.step_count}] Need new inference: {need_new_inference}")
            # ALWAYS send obs to seg server, but only request action when needed
            try:
                result = self.request_actions(obs, require_action=need_new_inference)
                
                # If we requested actions, process them
                if need_new_inference and result is not None:
                    action_seq = result['action']
                    
                    rospy.loginfo("#" * 60)
                    rospy.loginfo(f"Received {len(action_seq)} actions")
                    rospy.loginfo("#" * 60)
                    
                    if mode == "close_loop":
                        self.action_queue.clear()
                        actions_executed_since_last_inference = 0
                    
                    for action in action_seq:
                        self.action_queue.append(action)
            
            except Exception as e:
                rospy.logerr(f"Failed to communicate with server: {e}")
                import traceback
                traceback.print_exc()
            
            # Execute action
            if len(self.action_queue) > 0:
                action = self.action_queue.popleft()
                success = self.apply_policy_action(action)
                print(f"[Stage {self.current_stage} | Step {self.step_count}] Executed action: pos_delta={action[:3]}, gripper={action[7]:.2f}, stop={action[8]:.4f}")
                
                if success:
                    actions_executed_since_last_inference += 1
                    
                    # Check termination condition
                    if self.check_termination_condition(action):
                        cprint("Policy completed successfully!", "green", attrs=["bold"])
                        return True
                    
                    if self.step_count % 10 == 0:
                        pos_delta = np.linalg.norm(action[:3])
                        rospy.loginfo(f"[Step {self.step_count}] pos_delta={pos_delta:.6f}, queue={len(self.action_queue)}")
                else:
                    rospy.logerr("Action failed, stopping")
                    break
            
            # rate.sleep()
        
        rospy.loginfo("Policy phase ended")
        return False
    
    
    def reset_for_new_stage(self):
        """Reset state for a new stage"""
        self.current_obj_pose = None
        self.last_tcp_pose = None
        self.action_queue.clear()
        self.termination_window.clear()
        self.step_count = 0
    
    
    def shutdown(self):
        """Cleanup"""
        rospy.loginfo("Shutting down...")
        self.rtde_c.stopScript()
        self.gripper.disconnect()
        self.seg_sock.close()
        self.action_sock.close()
        self.ctx.term()


# =====================================================================
# Main
# =====================================================================
def main():
    try:
        node = HybridControlNode()
        
        # Wait for observations
        rospy.loginfo("Waiting for observations...")
        while len(node.state_buf) < OBS_STEPS and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Observations ready!")
        
        num_stages = len(MODEL_PATH)
        
        for stage in range(num_stages):
            cprint(f"\n{'#'*60}", "magenta", attrs=["bold"])
            cprint(f"STARTING STAGE {stage}/{num_stages - 1}", "magenta", attrs=["bold"])
            cprint(f"{'#'*60}\n", "magenta", attrs=["bold"])
            
            # Set current stage
            node.current_stage = stage
            node.reset_for_new_stage()
            
            # Phase 1: Gradient field
            success = node.run_gradient_field_phase()
            
            # if not success:
            #     cprint("Gradient field phase failed", "red")
            #     return
            
            cprint("\nTransitioning to policy control in 2 seconds...", "yellow")
            rospy.sleep(2.0)
            
            # Phase 2: Policy control
            policy_success = node.run_policy_phase(mode="open_loop", actions_per_inference=10)
            # node.run_policy_phase(mode="close_loop", actions_per_inference=5)
            
            if policy_success:
                cprint(f"\n{'='*60}", "green", attrs=["bold"])
                cprint(f"STAGE {stage} COMPLETED SUCCESSFULLY!", "green", attrs=["bold"])
                cprint(f"{'='*60}\n", "green", attrs=["bold"])
            else:
                cprint(f"\nStage {stage} policy phase did not complete successfully", "yellow")
            
            # # Transition delay between stages (except after last stage)
            # if stage < num_stages - 1:
            #     cprint(f"\nTransitioning to stage {stage + 1} in 2 seconds...", "yellow")
            #     rospy.sleep(2.0)
        
            node.rtde_c.moveJ(INITIAL_JOINT, speed=0.5, acceleration=0.25)
            if END_OPEN_GRIPPER[stage]:
                cprint("Opening gripper at end of stage", "yellow")
                node.gripper.move(3, 155, 155)
            rospy.sleep(2.0)
        
        cprint("\n" + "="*60, "green", attrs=["bold"])
        cprint("ALL STAGES COMPLETED!", "green", attrs=["bold"])
        cprint("="*60 + "\n", "green", attrs=["bold"])
        
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