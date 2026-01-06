import numpy as np
import open3d as o3d

# 加载点云数据
file_path = "/home/mainuser/UR5_Policy/data_recording/data_recording/data_cubeoncube/episode_1/point_cloud_hand.npy"
point_cloud_data = np.load(file_path)  # shape should be (N, 6)

# 拆分 xyz 和 rgb
points = point_cloud_data[:, :3]
colors = point_cloud_data[:, 3:]

points = points[:, :3]

# 确保是 float64 类型
points = points.astype(np.float64)

# 转换为 Open3D 格式
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])
