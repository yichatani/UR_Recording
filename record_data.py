import time
import json
import numpy as np
import pyrealsense2 as rs
import os
import threading
import robotiq_gripper
from pynput import keyboard 
from rtde_receive import RTDEReceiveInterface 
import shutil
import pyk4a
from pyk4a import Config, PyK4A
import sys
import cv2
from PIL import Image

## Script to collect demostrations of task for robot 

## Press 'c' to start recording new episode 
## Press 's' to stop recording 
## Press 'd' to delete most recent episode (only when not recording)
## Press 'q' to quit 

class Data(): 
    def __init__(self): 
        self.ROBOT_HOST = "192.168.56.101" 
        self.num_points = 1024 # Adjust number of points to downsample to here
        self.state_shape = 7 # Adjust state shape here
        self.action_shape = 7 # Adjust action state here

        # Params for hand camera
        self.width = 640
        self.height = 480
        self.fps = 30

        # start robot and camera
        self.k4a = self.initialize_camera()
        self.realsense = self.initialize_hand_camera()
        self.rtde_r, self.gripper = self.initialize_robot()

        # Params for point cloud processing
        # self.bbox = [-500, -400, -600, 1000, 200, 1500]
        self.bbox = [-500, -400, -600, 1000, 200, 1000]
        self.plane = [0.00, 0.42, 0.91, -538.08]
        self.distance_threshold = 10
        self.handbbox = [-300, -300, -300, 300, 300, 100]

        # global data variables
        global color_array, depth_array, point_cloud_array, color_hand_array, depth_hand_array, point_cloud_hand_array, point_concatenate_array
        global joint_state_array, joint_action_array, eef_state_array, eef_action_array, eef_force_array
        color_array, depth_array, point_cloud_array, color_hand_array, depth_hand_array, point_cloud_hand_array, point_concatenate_array = [], [], [], [], [], [], []
        joint_state_array, joint_action_array, eef_state_array, eef_action_array, eef_force_array = [], [], [], [], []

    def initialize_camera(self): 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                # camera_fps=pyk4a.FPS.FPS_30,
                camera_fps=pyk4a.FPS.FPS_15,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only= True,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        
        return k4a

    def initialize_hand_camera(self):
        realsense = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = realsense.start(config)
        
        # 创建 align 对象，将 depth 对齐到 color
        self.align_to_color = rs.align(rs.stream.color)
        
        return realsense

    def initialize_robot(self): 
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        print("Creating gripper...")
        gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        gripper.connect(self.ROBOT_HOST, 63352)
        print("Activating gripper...")
        gripper.activate()
        return rtde_r, gripper

    # Function to get robot joint state
    def get_robot_joint_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get robot eef state
    def get_robot_eef_state(self):
        state = np.array(self.rtde_r.getActualTCPPose())
        action = np.array(self.rtde_r. getTargetTCPPose())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get end-effector force
    def get_robot_eef_force(self):
        force = np.array(self.rtde_r.getActualTCPForce())
        return force

    #-----------------------------------------------------------------------------------------------------------#
    # Function to get visual observations (RGB-D & PC) 
    def get_visual_obs(self):
        capture = self.k4a.get_capture()
        capture_hand = self.realsense.wait_for_frames()

        if (
            capture is not None
            and capture.color is not None
            and capture.depth is not None
        ):
            # RealSense 对齐
            aligned_frames = self.align_to_color.process(capture_hand)
            
            if aligned_frames is None or aligned_frames.get_color_frame() is None or aligned_frames.get_depth_frame() is None:
                raise ValueError("RealSense aligned frames is None.")
            
            # Kinect RGB
            color_image = capture.color[:, :, :3]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Kinect Depth - 使用 transformed_depth_image 直接获取对齐后的深度图
            depth_image = capture.transformed_depth

            # Hand RGB (已对齐)
            color_hand_image = np.asarray(
                aligned_frames.get_color_frame().get_data()
            )
            color_hand_image = cv2.cvtColor(
                color_hand_image, cv2.COLOR_BGR2RGB
            )

            # Hand Depth (已对齐)
            depth_hand_image = np.asarray(
                aligned_frames.get_depth_frame().get_data()
            )

            return color_image, depth_image, color_hand_image, depth_hand_image
        else:
            raise ValueError("Kinect or RealSense capture is None.")



    # Function to handle key presses
    def on_press(self, key):
        global recording, current_time

        try:
            if key.char == 'c':
                if not recording:
                    current_time = time.strftime("%Y%m%d%H%M%S")
                    recording = True # start synchronization
                    print(f"Started recording episode ...")

            elif key.char == 's':
                if recording:
                    recording = False
                    episode_folder = os.path.join(data_folder, current_time)
                    print(f"Stopped recording episode ...")

                    time.sleep(1)
                    self.save_data(episode_folder)
                    print("Data saved")

                    time.sleep(1)
                    self.save_instruction(episode_folder)
                    print('Language instructions saved')

            elif key.char == 'q':
                if recording:
                    recording = False
                print("Quitting session...")
                return False  # Stop listener

            elif key.char == 'd':
                if not recording:
                    confirmation = input("Are you sure you want to delete the most recent episode? (y/n):")
                    if 'y' in confirmation:
                        # Delete the episode folder
                        episode_folder = os.path.join(data_folder, current_time)
                        if os.path.exists(episode_folder):
                            shutil.rmtree(episode_folder)
                        print(f"Deleted episode {episode_folder[-14:]}")
                    else:
                        print("Deletion canceled.")
                    

        except AttributeError:
            pass


    # Synchronization mechanism
    def synchronized_capture(self, frequency=5):
        global recording

        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                # Get visual observations
                # point_cloud, color_image, depth_image, color_hand_image, depth_hand_image,point_cloud_hand, point_concatenate = self.get_visual_obs()
                color_image, depth_image, color_hand_image, depth_hand_image = self.get_visual_obs()


                # Get robot state
                joint_state, joint_action = self.get_robot_joint_state()
                eef_state, eef_action = self.get_robot_eef_state()
                eef_force = self.get_robot_eef_force()

                if joint_state is not None and eef_state is not None:
                    # Create episode folder if it doesn't exist
                    episode_folder = os.path.join(data_folder, current_time)
                    if not os.path.exists(episode_folder):
                        os.makedirs(episode_folder)

                    # Save data
                    self.synchronize_data(color_image, depth_image, color_hand_image, depth_hand_image, \
                                          joint_state, joint_action, eef_state, eef_action, eef_force)
                    print("loading")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def synchronize_data(self, color_image, depth_image, color_hand_image, depth_hand_image,joint_state, joint_action, eef_state, eef_action, eef_force):
        # visual observations
        color_array.append(color_image)
        depth_array.append(depth_image)
        color_hand_array.append(color_hand_image)
        depth_hand_array.append(depth_hand_image)
            
        # states & actions
        joint_state_array.append(joint_state)
        joint_action_array.append(joint_action)
        eef_state_array.append(eef_state)
        eef_action_array.append(eef_action)
        eef_force_array.append(eef_force)

    def save_data(self, episode_folder):
        # Define file paths
        color_file = os.path.join(episode_folder, "rgb.npy")
        depth_file = os.path.join(episode_folder, "depth.npy")

        color_hand_file = os.path.join(episode_folder, "rgb_hand.npy")
        depth_hand_file = os.path.join(episode_folder, "depth_hand.npy")

        joint_state_file = os.path.join(episode_folder, "joint_state.npy")
        joint_action_file = os.path.join(episode_folder, "joint_action.npy")
        eef_state_file = os.path.join(episode_folder, "eef_state.npy")
        eef_action_file = os.path.join(episode_folder, "eef_action.npy")
        eef_force_file = os.path.join(episode_folder, "eef_force.npy")

        # List -> array
        color_image = np.array(color_array)
        depth_image = np.array(depth_array)
        color_hand_image = np.array(color_hand_array)
        depth_hand_image = np.array(depth_hand_array)

        joint_state = np.array(joint_state_array)
        joint_action = np.array(joint_action_array)
        eef_state = np.array(eef_state_array)
        eef_action = np.array(eef_action_array)
        eef_force = np.array(eef_force_array)

        # Save data into npy format
        np.save(color_file, color_image)
        np.save(depth_file, depth_image)
        np.save(color_hand_file, color_hand_image)
        np.save(depth_hand_file, depth_hand_image)

        np.save(joint_state_file, joint_state)
        np.save(joint_action_file, joint_action)
        np.save(eef_state_file, eef_state)
        np.save(eef_action_file, eef_action)
        np.save(eef_force_file, eef_force)

        # Save language instruction into json format
        instruction_json = {"instruction": instruction}
        with open(os.path.join(episode_folder, "instruction.json"), "w") as f:
            json.dump(instruction_json, f, indent=4)
  
        color_array.clear()             
        depth_array.clear()            
        color_hand_array.clear()       
        depth_hand_array.clear()        
        joint_state_array.clear()       
        joint_action_array.clear()      
        eef_state_array.clear()         
        eef_action_array.clear()        
        eef_force_array.clear()

    def save_instruction(self, episode_folder):
        """Save language instruction - you may need to implement this if it's a separate method"""
        pass

def main():
    # Global variables for controlling the recording state
    global data_folder, recording, instruction
    data_folder = "/home/ani/UR_data_recording/data"

    recording = False
    instruction = input("Type the instruction of this recording capture: ")

    data = Data()

    print("Start Recording, Press C to start recording episode")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=data.synchronized_capture)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=data.on_press)
    listener.start()
    listener.join()

    print("Stopped synchronized capture.")


if __name__ == '__main__':
    main()