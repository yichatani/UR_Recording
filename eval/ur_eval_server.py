#!/usr/bin/env python3
"""
UR Real Robot Policy Control Script
- Collects observations (RGB, Depth, State) from ROS topics
- Sends to remote policy server via ZMQ
- Receives actions (9D: 8D motion + 1D stop label)
- Executes actions using RTDE control
"""

import sys
import os
import rospy
import numpy as np
import cv2
import zmq
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import robotiq_gripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# =====================================================================
# Config
# =====================================================================
ROBOT_HOST = "192.168.56.101"

# Policy server addresses
SEG_ADDR = "tcp://192.168.56.55:5556"      # Send obs to seg server
ACTION_ADDR = "tcp://192.168.56.60:5557"   # Receive action from policy

# Control parameters
OBS_STEPS = 1      # Number of observation history frames
IMG_STEPS = 1      # Number of image history frames
CONTROL_HZ = 30    # Control frequency (Hz)
# ACTION_SCALE = 1.0  # Scale factor for action execution

# Stop detection parameters
STOP_THRESHOLD = 0.5         # Threshold for stop signal
CUMULATIVE_STOP_COUNT = 2    # How many times to detect stop before stopping

# =====================================================================
# Utility Functions
# =====================================================================
def rotvec_to_quat_xyzw(rx, ry, rz):
    """Convert rotation vector to quaternion [qx, qy, qz, qw]"""
    rotvec = np.array([rx, ry, rz])
    quat_xyzw = R.from_rotvec(rotvec).as_quat()
    return quat_xyzw


def quat_xyzw_to_rotvec(qx, qy, qz, qw):
    """Convert quaternion [qx, qy, qz, qw] to rotation vector"""
    quat_xyzw = np.array([qx, qy, qz, qw])
    rotvec = R.from_quat(quat_xyzw).as_rotvec()
    return rotvec


def ensure_quaternion_continuity(quat, last_quat):
    """Ensure quaternion continuity by avoiding q and -q jumps"""
    if last_quat is None:
        return quat
    
    dot_product = np.dot(quat, last_quat)
    
    if dot_product < 0:
        quat = -quat
    
    return quat


# =====================================================================
# Policy Control Node
# =====================================================================
class PolicyControlNode:
    def __init__(self):
        # ROS node
        rospy.init_node("ur_policy_control", anonymous=True)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Robot interfaces
        rospy.loginfo("Connecting to robot...")
        self.rtde_c = RTDEControlInterface(ROBOT_HOST)
        self.rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        
        # Gripper
        rospy.loginfo("Connecting to gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(ROBOT_HOST, 63352)
        self.gripper.activate()
        
        # ZMQ sockets
        self.ctx = zmq.Context.instance()
        
        # Connect to seg server
        self.seg_sock = self.ctx.socket(zmq.REQ)
        self.seg_sock.connect(SEG_ADDR)
        rospy.loginfo(f"Connected to SegServer at {SEG_ADDR}")
        
        # Connect to policy server for actions
        self.action_sock = self.ctx.socket(zmq.REQ)
        self.action_sock.connect(ACTION_ADDR)
        rospy.loginfo(f"Connected to PolicyServer at {ACTION_ADDR}")
        
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
        
        rospy.loginfo("Policy control node initialized")
        rospy.loginfo(f"Stop threshold: {STOP_THRESHOLD}, cumulative count: {CUMULATIVE_STOP_COUNT}")
    
    
    def observation_callback(self, wrist_color_msg, wrist_depth_msg):
        """Process incoming observations"""
        # Convert images
        wrist_img = self.bridge.imgmsg_to_cv2(wrist_color_msg, "bgr8")
        wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        wrist_depth = self.bridge.imgmsg_to_cv2(wrist_depth_msg)
        
        # Get robot state
        eef_pose = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        quat_xyzw = rotvec_to_quat_xyzw(eef_pose[3], eef_pose[4], eef_pose[5])
        quat_xyzw = ensure_quaternion_continuity(quat_xyzw, self.last_quat)
        self.last_quat = quat_xyzw.copy()
        
        gripper_width = self.gripper.get_current_position()
        
        # Build 8D state: [x, y, z, qx, qy, qz, qw, width]
        state = np.concatenate([eef_pose[:3], quat_xyzw, [gripper_width]]).astype(np.float32)
        
        # Update buffers
        self.state_buf.append(state)
        self.img_buf.append(wrist_img)
        self.depth_buf.append(wrist_depth)
    
    
    def get_observation(self):
        """Get stacked observation from buffers"""
        if len(self.state_buf) < OBS_STEPS or len(self.img_buf) < IMG_STEPS or len(self.depth_buf) < IMG_STEPS:
            return None
        
        obs = {
            "state": np.stack(self.state_buf, axis=0),      # (OBS_STEPS, 8)
            "image": np.stack(self.img_buf, axis=0),        # (IMG_STEPS, H, W, 3)
            "depth": np.stack(self.depth_buf, axis=0),      # (IMG_STEPS, H, W)
        }
        return obs
    
    
    def request_actions(self, obs):
        """Request actions from policy server"""
        # Send obs to seg server
        self.seg_sock.send_pyobj(obs)
        self.seg_sock.recv()
        
        # Receive result from policy server
        result = self.action_sock.recv_pyobj()
        self.action_sock.send(b"ok")
        
        return result
    
    
    def apply_action(self, action):
        """Apply single action step (9D: 8D motion + 1D stop label)
        
        Action format: [dx, dy, dz, dqx, dqy, dqz, dqw, gripper_width, stop_label]
        - action[:3]: position delta (m)
        - action[3:7]: quaternion delta (xyzw format)
        - action[7]: gripper width ABSOLUTE value (m)
        - action[8]: stop label (0-1, not used here)
        """
        # Get current pose
        current_pose = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        current_pos = np.array(current_pose[:3])
        current_quat = rotvec_to_quat_xyzw(current_pose[3], current_pose[4], current_pose[5])
        
        # Apply position delta
        # target_pos = current_pos + action[:3] * ACTION_SCALE
        target_pos = current_pos + action[:3]
        
        # Apply quaternion delta
        current_rot = R.from_quat(current_quat)
        delta_rot = R.from_quat(action[3:7])
        target_rot = current_rot * delta_rot
        target_rotvec = target_rot.as_rotvec()
        
        # Build target pose
        target_pose = np.concatenate([target_pos, target_rotvec])
        
        # Execute motion
        try:
            self.rtde_c.moveL(target_pose.tolist(), speed=0.1, acceleration=0.3)
            
            # Set gripper
            gripper_width = action[7]
            self.gripper.move(gripper_width, 155, 255)
            # if gripper_width < 3:
            #     self.gripper.move(0, 155, 255)  # Close
            # else:
            #     # target_position = int(gripper_width * 255 / 0.085)  # Normalize to [0, 255]
            #     self.gripper.move(gripper_width, 255, 255)
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to execute action: {e}")
            return False
    
    
    def run_control_loop(self, mode="open_loop", actions_per_inference=10):
        """Main control loop
        
        Args:
            mode: "open_loop" or "closed_loop"
            actions_per_inference: for closed_loop mode, how many actions to execute before re-inference
        """
        rospy.loginfo(f"Starting control loop in {mode} mode")
        
        rate = rospy.Rate(CONTROL_HZ)
        actions_executed_since_last_inference = 0
        
        while not rospy.is_shutdown():
            self.step_count += 1
            
            # Check if we need new actions
            need_new_inference = False
            
            if mode == "open_loop":
                # Request new actions when queue is empty
                need_new_inference = (len(self.action_queue) == 0)
            elif mode == "closed_loop":
                # Request new actions when queue is empty OR executed enough actions
                need_new_inference = (
                    len(self.action_queue) == 0 or
                    actions_executed_since_last_inference >= actions_per_inference
                )
            
            # Request new actions if needed
            if need_new_inference:
                obs = self.get_observation()
                
                if obs is not None:
                    rospy.loginfo(f"[Step {self.step_count}] Requesting new actions...")
                    
                    try:
                        result = self.request_actions(obs)
                        
                        action_seq = result['action']  # (horizon, 9)
                        
                        rospy.loginfo("#" * 60)
                        rospy.loginfo(f"Received action sequence shape: {action_seq.shape}")
                        rospy.loginfo("#" * 60)
                        
                        # For closed loop, clear old actions
                        if mode == "closed_loop":
                            self.action_queue.clear()
                            actions_executed_since_last_inference = 0
                        
                        # Add new actions to queue
                        for action in action_seq:
                            self.action_queue.append(action)
                        
                        rospy.loginfo(f"Added {len(action_seq)} actions to queue")
                        
                        # # Check stop signal (from action[:, 8])
                        # # We use the first action's stop label as the stop signal
                        # if action_seq.shape[0] > 0:
                        #     stop_signal = action_seq[0, 8]
                            
                        #     if stop_signal > STOP_THRESHOLD:
                        #         self.total_stop_count += 1
                                
                        #         rospy.logwarn(f"Stop signal detected!")
                        #         rospy.logwarn(f"  Signal value: {stop_signal:.4f} > {STOP_THRESHOLD}")
                        #         rospy.logwarn(f"  Total stop count: {self.total_stop_count}/{CUMULATIVE_STOP_COUNT}")
                                
                        #         if self.total_stop_count >= CUMULATIVE_STOP_COUNT:
                        #             rospy.logwarn(f"Reached {CUMULATIVE_STOP_COUNT} stop detections")
                        #             rospy.logwarn("Stopping control loop")
                        #             break
                        #     else:
                        #         rospy.loginfo(f"Stop signal below threshold: {stop_signal:.4f} <= {STOP_THRESHOLD}")
                    
                    except Exception as e:
                        rospy.logerr(f"Failed to get actions: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Execute action from queue
            if len(self.action_queue) > 0:
                action = self.action_queue.popleft()
                success = self.apply_action(action)
                
                if success:
                    actions_executed_since_last_inference += 1
                    
                    if self.step_count % 10 == 0:
                        pos_delta = np.linalg.norm(action[:3])
                        stop_label = action[8]
                        rospy.loginfo(f"[Step {self.step_count}] Executed action: "
                                    f"pos_delta={pos_delta:.6f}, stop={stop_label:.4f}, "
                                    f"queue={len(self.action_queue)}")
                else:
                    rospy.logerr("Action execution failed, stopping")
                    break
            
            rate.sleep()
        
        rospy.loginfo("Control loop ended")
        rospy.loginfo(f"Final stats: {self.step_count} steps, {self.total_stop_count} stop detections")
    
    
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
        node = PolicyControlNode()
        
        # Wait for observations to accumulate
        rospy.loginfo("Waiting for observations...")
        while len(node.state_buf) < OBS_STEPS and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Observations ready, starting control loop")
        
        # Choose control mode
        # control_mode = "closed_loop"
        control_mode = "open_loop"
        actions_per_inference = 10
        
        node.run_control_loop(mode=control_mode, actions_per_inference=actions_per_inference)
        
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