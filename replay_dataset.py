#!/usr/bin/env python3
"""
Replay recorded LeRobot dataset on UR5e robot using position control
"""
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import robotiq_gripper
from scipy.spatial.transform import Rotation
import time
from pynput import keyboard

# Robot parameters
ROBOT_HOST = "192.168.56.101"
DATA_ROOT = "/home/ani/UR_Recording/data/sid_pour_water"
REPO_ID = "ani/sid"

# Replay parameters
REPLAY_SPEED = 1.0  # 1.0 = normal speed, 0.5 = half speed
TARGET_FPS = 30

class DatasetReplayer:
    def __init__(self, robot_host, dataset_root, repo_id):
        self.dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        
        # Initialize robot interfaces
        print("Connecting to robot...")
        self.rtde_c = RTDEControlInterface(robot_host)
        self.rtde_r = RTDEReceiveInterface(robot_host)
        
        # Initialize gripper
        print("Initializing gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(robot_host, 63352)
        self.gripper.activate()
        
        self.is_playing = False
        self.should_quit = False
        
    def quat_to_rotvec(self, qx, qy, qz, qw):
        """Convert quaternion (xyzw) to rotation vector for UR"""
        quat_xyzw = np.array([qx, qy, qz, qw])
        rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
        return rotvec
    
    def replay_episode(self, episode_id):
        """Replay a single episode using position control"""
        # Get all frames for this episode
        ep_col = np.asarray(self.dataset.hf_dataset["episode_index"])
        fr_col = np.asarray(self.dataset.hf_dataset["frame_index"])
        
        idxs = np.where(ep_col == episode_id)[0]
        idxs = idxs[np.argsort(fr_col[idxs])]
        
        print(f"\n{'='*50}")
        print(f"Episode {episode_id}: {len(idxs)} frames")
        print(f"{'='*50}")
        
        if len(idxs) == 0:
            print(f"Episode {episode_id} not found!")
            return
        
        # Move to start pose
        print("Moving to start pose...")
        first_sample = self.dataset[int(idxs[0])]
        start_action = first_sample["action"].detach().cpu().numpy().astype(np.float32)
        
        # Extract pose: [x, y, z, qx, qy, qz, qw, width]
        pos = start_action[:3]
        quat_xyzw = start_action[3:7]
        gripper_width = start_action[7]
        
        # Convert quaternion to rotation vector
        rotvec = self.quat_to_rotvec(*quat_xyzw)
        target_pose = np.concatenate([pos, rotvec])
        
        # Move to start with moveL
        self.rtde_c.moveL(target_pose.tolist(), 0.25, 0.5)  # ✅ 位置参数
        self.gripper.move(int(gripper_width), 155, 255)
        time.sleep(1.0)
        
        print("Press 'p' to start replay, 'space' to pause/resume, 'q' to quit")
        
        # Wait for user to press 'p'
        while not self.is_playing and not self.should_quit:
            time.sleep(0.1)
        
        if self.should_quit:
            return
        
        print("▶ Starting replay...")
        
        # ✅ servoL 参数: (pose, velocity, acceleration, dt, lookahead_time, gain)
        dt = 1.0 / (TARGET_FPS * REPLAY_SPEED)
        velocity = 0.5        # 最大速度
        acceleration = 1.2    # 加速度
        lookahead_time = 0.1  # 前瞻时间
        gain = 300            # 位置增益
        
        for i, global_i in enumerate(idxs):
            if self.should_quit:
                break
                
            # Pause handling
            while not self.is_playing and not self.should_quit:
                self.rtde_c.servoStop()
                time.sleep(0.1)
            
            if self.should_quit:
                break
            
            sample = self.dataset[int(global_i)]
            action = sample["action"].detach().cpu().numpy().astype(np.float32)
            
            # Extract target pose: [x, y, z, qx, qy, qz, qw, width]
            pos = action[:3]
            quat_xyzw = action[3:7]
            gripper_width = action[7]
            
            # Convert quaternion to rotation vector
            rotvec = self.quat_to_rotvec(*quat_xyzw)
            target_pose = np.concatenate([pos, rotvec])
            
            # ✅ 使用位置参数调用 servoL
            self.rtde_c.servoL(
                target_pose.tolist(),  # pose
                velocity,              # velocity
                acceleration,          # acceleration
                dt,                    # dt
                lookahead_time,        # lookahead_time
                gain                   # gain
            )
            
            # Control gripper
            current_gripper = self.gripper.get_current_position()
            if abs(gripper_width - current_gripper) > 5:
                self.gripper.move(int(gripper_width), 155, 255)
            
            # Progress info
            if i % 10 == 0:
                current_pose = self.rtde_r.getActualTCPPose()
                pos_error = np.linalg.norm(pos - current_pose[:3])
                print(f"Frame {i+1}/{len(idxs)} | Target: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                      f"Error: {pos_error*1000:.1f}mm | Gripper: {gripper_width:.1f}")
            
            time.sleep(dt)
        
        # Stop servo control
        self.rtde_c.servoStop()
        print("✅ Replay completed")
    
    def on_press(self, key):
        """Keyboard handler"""
        try:
            if key.char == 'p':
                if not self.is_playing:
                    self.is_playing = True
                    print("▶ Playing")
                    
            elif key.char == 'q':
                self.should_quit = True
                print("Quitting...")
                return False
                
            elif key == keyboard.Key.space:
                self.is_playing = not self.is_playing
                if self.is_playing:
                    print("▶ Resumed")
                else:
                    print("⏸ Paused")
                    
        except AttributeError:
            pass
    
    def run(self):
        """Main run loop"""
        print(f"\nDataset: {self.dataset.num_episodes} episodes")
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        try:
            while not self.should_quit:
                episode_id = input("\nEnter episode ID to replay (or 'q' to quit): ")
                
                if episode_id.lower() == 'q':
                    break
                
                try:
                    ep_id = int(episode_id)
                    if ep_id >= self.dataset.num_episodes:
                        print(f"Episode {ep_id} does not exist! Max: {self.dataset.num_episodes - 1}")
                        continue
                    
                    # Reset state
                    self.is_playing = False
                    self.should_quit = False
                    
                    self.replay_episode(ep_id)
                    
                except ValueError:
                    print("Invalid input! Please enter a number.")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            listener.stop()
            print("Replay session ended")


def main():
    replayer = DatasetReplayer(ROBOT_HOST, DATA_ROOT, REPO_ID)
    replayer.run()


if __name__ == "__main__":
    main()