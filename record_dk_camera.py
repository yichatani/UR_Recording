#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
import numpy as np
from pynput import keyboard

print("Starting Kinect DK High-Res Recorder...")

bridge = CvBridge()
ffmpeg_process = None
recording = True
frame_count = 0

# 视频设置
VIDEO_OUTPUT = "/home/ani/UR_Recording/kwj/kinect_recording2.mp4"
FPS = 30
WIDTH, HEIGHT = 1920, 1080


def start_ffmpeg():
    global ffmpeg_process
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖已有文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',  # 从 stdin 读取
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        VIDEO_OUTPUT
    ]
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rospy.loginfo(f"Started ffmpeg recording to {VIDEO_OUTPUT}")


def color_callback(msg):
    global ffmpeg_process, frame_count
    
    if not recording or ffmpeg_process is None:
        return
    
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Resize if needed
    if img.shape[:2] != (HEIGHT, WIDTH):
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    try:
        ffmpeg_process.stdin.write(img.tobytes())
        frame_count += 1
        if frame_count % 100 == 0:
            rospy.loginfo(f"Recorded {frame_count} frames")
    except BrokenPipeError:
        rospy.logwarn("FFmpeg pipe broken")


def on_press(key):
    global recording, ffmpeg_process
    
    try:
        if key.char == 'q':
            rospy.loginfo("Stopping recording...")
            recording = False
            if ffmpeg_process is not None:
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()
                rospy.loginfo(f"Video saved: {VIDEO_OUTPUT}, total {frame_count} frames")
            return False
    except AttributeError:
        pass


def main():
    rospy.init_node("kinect_hires_recorder", anonymous=True)
    
    start_ffmpeg()
    
    rospy.Subscriber("/rgb/image_raw", Image, color_callback, queue_size=5)
    
    rospy.loginfo("Kinect DK High-Res Recorder started")
    rospy.loginfo("Recording continuously... Press 'q' to quit and save")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()
    
    rospy.loginfo("Program terminated.")


if __name__ == "__main__":
    main()