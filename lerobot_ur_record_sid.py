#!/usr/bin/env python3
import rospy
import math
from scipy.spatial.transform import Rotation
import robotiq_gripper
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
import cv2
import time
from pynput import keyboard
import shutil
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rtde_receive import RTDEReceiveInterface

print("Starting subscriber_ur_record.py...")

recording = False
bridge = CvBridge()
instruction = None
last_delete_time = 0
dataset = None
episode_recording = False
last_quat = None  # ✅ 用于跟踪上一帧四元数


def initialize_dataset(root_dir, save_depth=True):
    from pathlib import Path
    import shutil

    root_dir = Path(root_dir)
    repo_id = "ani/sid"

    ee_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "width"]

    features_dict = {
        "rgb_wrist": {"dtype": "video", "shape": (540, 960, 3), "names": ["height", "width", "channel"]},
        # "rgb_global": {"dtype": "video", "shape": (1536, 2048, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ee_names},
        "action": {"dtype": "float32", "shape": (8,), "names": ee_names},
    }
    if save_depth:
        # features_dict["depth_wrist"]  = {"dtype": "float32", "shape": (540, 960), "names": ["height", "width"]}
        features_dict["depth_wrist"]  = {"dtype": "uint16", "shape": (540, 960), "names": ["height", "width"]}

    if root_dir.exists() and not (root_dir / "meta" / "info.json").exists():
        shutil.rmtree(root_dir)

    if (root_dir / "meta" / "info.json").exists():
        dataset = LeRobotDataset(repo_id=repo_id, root=str(root_dir))
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root_dir,
            robot_type="ur5e",
            fps=30,
            features=features_dict,
            use_videos=True,
        )

    dataset.start_image_writer(num_processes=2, num_threads=12)
    return dataset



ROBOT_HOST = "192.168.56.101"
rtde_r = RTDEReceiveInterface(ROBOT_HOST)
print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ROBOT_HOST, 63352)
print("Activating gripper...")
gripper.activate()


def rotvec_to_quat_xyzw(rx, ry, rz):
    rotvec = np.array([rx, ry, rz])
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return quat_xyzw


def ensure_quaternion_continuity(quat, last_quat):
    """
    ✅ 确保四元数连续性,避免 q 和 -q 跳变
    如果当前四元数与上一帧的点积为负,则翻转符号
    """
    if last_quat is None:
        return quat
    
    # 计算点积
    dot_product = np.dot(quat, last_quat)
    
    # 如果点积为负,翻转当前四元数
    if dot_product < 0:
        quat = -quat
    
    return quat


def callback(wrist_color_msg, wrist_depth_msg):
    global episode_recording, dataset, instruction, last_quat
    
    if not episode_recording or dataset is None:
        return

    # 转换图像
    wrist_img = bridge.imgmsg_to_cv2(wrist_color_msg, "bgr8")
    # wrist_img = cv2.resize(wrist_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    wrist_depth = bridge.imgmsg_to_cv2(wrist_depth_msg)
    # wrist_depth = cv2.resize(wrist_depth, (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    
    # global_img = bridge.imgmsg_to_cv2(global_color_msg, "bgr8")
    # # global_img = cv2.resize(global_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # global_img = cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # 获取机器人状态
    eef_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
    quat_xyzw = rotvec_to_quat_xyzw(eef_pose[3], eef_pose[4], eef_pose[5])
    
    quat_xyzw = ensure_quaternion_continuity(quat_xyzw, last_quat)
    last_quat = quat_xyzw.copy()
    
    gripper_width = gripper.get_current_position()
    
    # 构建8维state: [x,y,z, qx,qy,qz,qw, width]
    state = np.concatenate([eef_pose[:3], quat_xyzw, [gripper_width]]).astype(np.float32)

    # 构建frame字典
    frame = {
        "rgb_wrist": wrist_img,
        "depth_wrist": wrist_depth,
        "observation.state": state,
        "action": state,
    }
    dataset.add_frame(frame, task=instruction)
    
    rospy.loginfo(f"Recording... collected {dataset.episode_buffer['size']} frames")


def on_press(key):
    global episode_recording, last_delete_time, dataset, instruction, last_quat
    
    try:
        if key.char == 'c':
            if not episode_recording:
                episode_recording = True
                last_quat = None  # ✅ 重置四元数跟踪
                rospy.loginfo(f"▶ Started recording episode_{dataset.num_episodes}")

        elif key.char == 's':
            if episode_recording:
                episode_recording = False

                n = dataset.episode_buffer.get("size", 0)
                if n == 0:
                    rospy.logwarn("⚠️ No frames recorded in this episode, skip saving.")
                    return

                episode_idx = dataset.num_episodes
                rospy.loginfo(f"⏹ Stopped recording, saving episode_{episode_idx}...")
                
                dataset.save_episode()
                last_quat = None  # ✅ 重置四元数跟踪
                rospy.loginfo(f"✅ Episode {episode_idx} saved")
                rospy.loginfo(f"Next episode index: {dataset.num_episodes}")

        elif key.char == 'd':
            now = time.time()
            if now - last_delete_time < 1.0:
                if dataset.num_episodes > 0:
                    last_episode_idx = dataset.num_episodes - 1
                    
                    data_file = dataset.root / dataset.meta.get_data_file_path(last_episode_idx)
                    if data_file.exists():
                        data_file.unlink()
                        rospy.loginfo(f"🗑 Deleted data file: {data_file}")
                    
                    for vid_key in dataset.meta.video_keys:
                        video_file = dataset.root / dataset.meta.get_video_file_path(last_episode_idx, vid_key)
                        if video_file.exists():
                            video_file.unlink()
                            rospy.loginfo(f"🗑 Deleted video file: {video_file}")
                    
                    for cam_key in dataset.meta.camera_keys:
                        img_dir = dataset._get_image_file_path(last_episode_idx, cam_key, 0).parent
                        if img_dir.exists():
                            shutil.rmtree(img_dir)
                    
                    rospy.loginfo(f"Deleted episode_{last_episode_idx}")
                    
                    dataset.stop_image_writer()
                    dataset = initialize_dataset(str(dataset.root), save_depth=True)
                else:
                    rospy.logwarn("No episode available to delete.")
            else:
                rospy.loginfo("Press 'd' again within 1 second to confirm deletion.")
            last_delete_time = now

        elif key.char == 'q':
            rospy.loginfo("Quitting session...")
            if dataset is not None:
                dataset.stop_image_writer()
            return False

    except AttributeError:
        pass


def main():
    global instruction, dataset
    
    rospy.init_node("rgb_pose_recorder", anonymous=True)

    instruction = input("Input language instruction: ")
    
    DATASET_ROOT = "/home/ani/UR_Recording/data/sid_open_canvas_box"  
    dataset = initialize_dataset(DATASET_ROOT, save_depth=True)
    rospy.loginfo(f"Starting new recording at episode_{dataset.num_episodes}")

    wrist_color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    wrist_depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [wrist_color_sub, wrist_depth_sub],
        queue_size=20,
        slop=0.05,
        allow_headerless=True
    )
    ts.registerCallback(callback)

    rospy.loginfo("RGB (wrist) + UR5e Pose Recorder started.")
    rospy.loginfo("Press 'c' start, 's' stop and save, 'q' quit, 'd' delete file.")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

    rospy.loginfo("Program terminated.")


if __name__ == "__main__":
    main()