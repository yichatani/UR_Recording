#!/usr/bin/env python3
import rospy
import robotiq_gripper
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import threading
from rtde_receive import RTDEReceiveInterface 
import time
from pynput import keyboard
import re
import shutil
import json
from datetime import datetime

wrist_color_array, global_color_array = [], []  # color_array → wrist_color_array
wrist_depth_array = []
eef_pose_array = []
wrist_color_ts_array, global_color_ts_array= [], []
recording = False
current_time = None
data_folder = "/home/ani/UR_data_recording/data"
bridge = CvBridge()
instruction = None
episode_num = 0
last_delete_time = 0

ROBOT_HOST = "192.168.56.101"

rtde_r = RTDEReceiveInterface(ROBOT_HOST)
print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ROBOT_HOST, 63352)
print("Activating gripper...")
gripper.activate()

def get_max_episode_num():
    if not os.path.exists(data_folder):
        return 0
    pattern = re.compile(r'episode_(\d+)')
    max_num = 0
    for item in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, item)):
            match = pattern.match(item)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    return max_num


def get_next_episode_num():
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return 0

    subfolders = [
        name for name in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, name))
    ]

    next_num = len(subfolders)
    rospy.loginfo(f"🧩 Detected {next_num} existing episodes. Next: episode_{next_num}")
    return next_num


def save_data(episode_folder):
    if not os.path.exists(episode_folder):
        os.makedirs(episode_folder)

    # 1️⃣ 保存主要数据文件（保持原有格式）
    wrist_rgb_file = os.path.join(episode_folder, "wrist_color.npy")
    wrist_depth_file = os.path.join(episode_folder, "wrist_depth.npy")
    global_rgb_file = os.path.join(episode_folder, "global_color.npy")
    eef_pose_file = os.path.join(episode_folder, "eef_pose.npy")

    np.save(wrist_rgb_file, np.array(wrist_color_array))
    np.save(wrist_depth_file, np.array(wrist_depth_array))
    np.save(global_rgb_file, np.array(global_color_array))
    np.save(eef_pose_file, np.array(eef_pose_array))

    # 2️⃣ 构建 meta.json 内容
    meta_data = {
        "instruction": instruction if instruction else "",
        "timesteps": {},
        "meta": {
            "num_frames": len(wrist_color_ts_array),
            "save_time": datetime.now().isoformat(),
            "data_version": "v1.1"
        }
    }

    # 将所有时间戳序列化成 float（秒）
    for i in range(len(wrist_color_ts_array)):
        meta_data["timesteps"][str(i)] = {
            "wrist_ts": wrist_color_ts_array[i].to_sec() if hasattr(wrist_color_ts_array[i], "to_sec") else float(wrist_color_ts_array[i]),
            "global_ts": global_color_ts_array[i].to_sec() if hasattr(global_color_ts_array[i], "to_sec") else float(global_color_ts_array[i]),
        }

    # 3️⃣ 保存 meta.json 文件
    with open(os.path.join(episode_folder, "meta.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    # 4️⃣ 清空缓存
    wrist_color_array.clear()
    wrist_depth_array.clear()
    global_color_array.clear()
    eef_pose_array.clear()
    wrist_color_ts_array.clear()
    global_color_ts_array.clear()

    rospy.loginfo(f"✅ Data saved to {episode_folder} (meta.json included)")


def callback(wrist_color_msg, wrist_depth_msg, global_color_msg):
    global recording
    if not recording:
        return

    global rtde_r, gripper

    wrist_img = bridge.imgmsg_to_cv2(wrist_color_msg, "bgr8")
    wrist_img = cv2.resize(wrist_img, (640, 384))
    wrist_ts = wrist_color_msg.header.stamp

    wrist_depth = bridge.imgmsg_to_cv2(wrist_depth_msg)

    global_img = bridge.imgmsg_to_cv2(global_color_msg, "bgr8")
    global_img = cv2.resize(global_img, (640, 384))
    global_ts = global_color_msg.header.stamp

    eef_pose = rtde_r.getActualTCPPose() # pose = [x, y, z, rx, ry, rz]
    # print(f"{eef_pose=}")

    gripper_width = gripper.get_current_position()
    # print(f"{gripper_width=}")
    eef_pose = np.concatenate([eef_pose, [gripper_width]])

    wrist_color_array.append(wrist_img)
    wrist_depth_array.append(wrist_depth)
    global_color_array.append(global_img)
    eef_pose_array.append(eef_pose)
    wrist_color_ts_array.append(wrist_ts)
    global_color_ts_array.append(global_ts)

    rospy.loginfo(f"Recording... collected {len(wrist_color_array)} frames")


def on_press(key):
    global recording, episode_num, last_delete_time
    try:
        if key.char == 'c':
            if not recording:
                recording = True
                rospy.loginfo(f"▶ Started recording episode_{episode_num}")

        elif key.char == 's':
            if recording:
                recording = False
                episode_folder = os.path.join(data_folder, f"episode_{episode_num}")
                rospy.loginfo(f"⏹ Stopped recording, saving episode_{episode_num}...")
                save_data(episode_folder)
                episode_num = get_next_episode_num()
                rospy.loginfo(f"Next episode index: {episode_num}")

        elif key.char == 'd':
            now = time.time()
            if now - last_delete_time < 1.0:
                if episode_num > 0:
                    last_episode_num = episode_num - 1
                    candidates = [f for f in os.listdir(data_folder) if f.startswith(f"episode_{last_episode_num}")]
                    if candidates:
                        folder_to_delete = os.path.join(data_folder, sorted(candidates)[-1])
                        shutil.rmtree(folder_to_delete)
                        rospy.loginfo(f"🗑 Deleted {folder_to_delete}")
                        episode_num = get_next_episode_num()
                    else:
                        rospy.logwarn(f"No folder found for episode_{last_episode_num}")
                else:
                    rospy.logwarn("No episode available to delete.")
            else:
                rospy.loginfo("Press 'd' again within 1 second to confirm deletion.")
            last_delete_time = now

        elif key.char == 'q':
            rospy.loginfo("Quitting session...")
            return False

    except AttributeError:
        pass


def main():
    global instruction, episode_num
    rospy.init_node("rgb_pose_recorder", anonymous=True)

    instruction = input("Input language instruction: ")
    episode_num = get_next_episode_num()
    rospy.loginfo(f"Starting new recording at episode_{episode_num}")

    wrist_color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    wrist_depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    global_color_sub = message_filters.Subscriber("/rgb/image_raw", Image)

    ts = message_filters.ApproximateTimeSynchronizer(
        [wrist_color_sub, wrist_depth_sub, global_color_sub],
        queue_size=20,
        slop=0.05,
        allow_headerless=True
    )
    ts.registerCallback(callback)

    rospy.loginfo("RGB (wrist + global) + Franka Pose Recorder started.")
    rospy.loginfo("Press 'c' start, 's' stop and save, 'q' quit, 'd' delete file.")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

    rospy.loginfo("Program terminated.")


if __name__ == "__main__":
    main()

