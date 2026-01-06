#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pprint import pprint

# =================== 配置 ===================
data_dir = "/home/ani/UR_data_recording/data/episode_2"

# =================== 加载函数 ===================
def load_data(data_dir):
    print(f"🔍 Checking episode folder: {data_dir}\n")

    # 1️⃣ 检查文件存在性
    required_files = [
        "wrist_color.npy",
        "wrist_depth.npy",
        "global_color.npy",
        "eef_pose.npy",
        "meta.json"
    ]
    for f in required_files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing file: {f}")
    print("✅ All required files found.\n")

    # 2️⃣ 读取数据
    wrist_imgs = np.load(os.path.join(data_dir, "wrist_color.npy"), allow_pickle=True)
    wrist_depths = np.load(os.path.join(data_dir, "wrist_depth.npy"), allow_pickle=True)
    global_imgs = np.load(os.path.join(data_dir, "global_color.npy"), allow_pickle=True)
    eef_pose = np.load(os.path.join(data_dir, "eef_pose.npy"))
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r"))

    print(f"📦 wrist_color shape: {wrist_imgs.shape}")
    print(f"📦 depth shape: {wrist_depths.shape}")
    print(f"📦 global_color shape: {global_imgs.shape}")
    print(f"📦 eef_pose shape:   {eef_pose.shape}")
    print(f"📜 Frames in meta:   {len(meta['timesteps'])}\n")

    return wrist_imgs, wrist_depths, global_imgs, eef_pose, meta


# =================== 时间戳检查 ===================
def analyze_timestamps(meta):
    timesteps = meta["timesteps"]
    wrist_ts = np.array([v["wrist_ts"] for v in timesteps.values()])
    global_ts = np.array([v["global_ts"] for v in timesteps.values()])

    # 基本统计
    print("🕒 Time Statistics:")
    print(f"  wrist:  {wrist_ts[0]:.3f} → {wrist_ts[-1]:.3f} (Δ={wrist_ts[-1]-wrist_ts[0]:.3f}s)")
    print(f"  global: {global_ts[0]:.3f} → {global_ts[-1]:.3f} (Δ={global_ts[-1]-global_ts[0]:.3f}s)")

    # 延迟分析
    delta_wrist_global = (wrist_ts - global_ts) * 1000  # ms
    print(f"⏱️  wrist-global mean offset: {np.mean(delta_wrist_global):.3f} ms ± {np.std(delta_wrist_global):.3f}")

    # 可视化
    plt.figure()
    plt.plot(delta_wrist_global, label="wrist - global (ms)")
    plt.legend()
    plt.xlabel("Frame index")
    plt.ylabel("Time offset (ms)")
    plt.title("Time Synchronization Offsets")
    plt.show()


# =================== 打印姿态与关节 ===================
def inspect_robot_data(eef_pose):
    print("🤖 Example Robot State:")
    for i in range(min(20, len(eef_pose))):  # 打印前三帧
        xyz = eef_pose[i, :3]
        quat = eef_pose[i, 3:6]
        width = eef_pose[i, 6]
        print(f"Frame {i}:")
        print(f"  pos = {xyz}")
        print(f"  quat = {quat}")
        print(f"  gripper width = {width:.4f}")


# =================== 可视化相机图像 ===================
def visualize_images(wrist_imgs, global_imgs):
    num_frames = len(wrist_imgs)
    print(f"🖼️  Visualizing {num_frames} frames (press ESC to quit)...")

    for i in range(num_frames):
        wrist = wrist_imgs[i]
        global_ = global_imgs[i]

        combined = np.hstack((
            cv2.putText(wrist.copy(), "Wrist", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2),
            cv2.putText(global_.copy(), "Global", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        ))

        cv2.imshow("Wrist (left) + Global (right)", combined)
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()



def visualize_wrist_image(wrist_imgs):
    num_frames = len(wrist_imgs)
    print(f"🖼️  Visualizing {num_frames} frames (press ESC to quit)...")

    for i in range(num_frames):
        wrist = wrist_imgs[i]
        # global_ = global_imgs[i]

        # combined = np.hstack((
        #     cv2.putText(wrist.copy(), "Wrist", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2),
        #     cv2.putText(global_.copy(), "Global", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # ))

        cv2.imshow("Wrist (left) + Global (right)", wrist)
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


def visualize_images_depths(wrist_imgs, depth_imgs):
    num_frames = len(wrist_imgs)
    print(f"🖼️  Visualizing {num_frames} frames (press ESC to quit)...")

    for i in range(num_frames):
        wrist = wrist_imgs[i]
        depth = depth_imgs[i]

        # combined = np.hstack((
        #     cv2.putText(wrist.copy(), "Wrist", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2),
        #     cv2.putText(depth.copy(), "Global", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # ))


        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # 伪彩色
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.imshow("Wrist (left) + Global (right)", depth_color)
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


# =================== 主函数 ===================
def main():
    wrist_imgs, wrist_depths, global_imgs, eef_pose, meta = load_data(data_dir)
    inspect_robot_data(eef_pose)
    analyze_timestamps(meta)
    visualize_images(wrist_imgs, global_imgs)
    # visualize_wrist_image(wrist_imgs)
    # visualize_images_depths(wrist_imgs, wrist_depths)
    print("✅ Done.")


if __name__ == "__main__":
    main()
