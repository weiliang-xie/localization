import rosbag
import utm
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-02-13-16-54-32.bag"
# bag_file = "mapping_2025-02-13-16-42-06.bag"
gnss_data = []
lidar_data = []
heading_data = []

def convert_gps_to_utm(lat, lon):
    """检查纬度范围并转换 UTM"""
    if not (-80.0 <= lat <= 84.0):
        return None, None
    try:
        utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
        return utm_x, utm_y
    except:
        return None, None

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == "/novatel718d/pos":
            utm_x, utm_y = convert_gps_to_utm(msg.latitude, msg.longitude)
            if utm_x is not None and utm_y is not None:
                gnss_data.append((msg.header.stamp.to_sec(), utm_x, utm_y, msg.altitude))

        elif topic == "/novatel718d/heading":
            # 解析航向（四元数）
            quat = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
            rot_mat = R.from_quat(quat).as_matrix()
            heading_data.append((msg.header.stamp.to_sec(), rot_mat))

        elif topic == "/lidar_points":
            timestamp = msg.header.stamp.to_sec()
            points = np.array([p[:3] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
            lidar_data.append((timestamp, points))

# **转换 LiDAR 点云到世界坐标系**
if len(gnss_data) == 0 or len(lidar_data) == 0:
    print("🚨 没有有效的 GNSS 或 LiDAR 数据")
    exit()

gnss_data = np.array(gnss_data)  # [time, x, y, z]
heading_data = dict(heading_data)  # 用字典存储，方便查找
lidar_transformed = []

for timestamp, points in lidar_data:
    if timestamp not in heading_data:
        continue  # 跳过没有匹配位姿的 LiDAR 帧

    # 获取位姿
    idx = np.searchsorted(gnss_data[:, 0], timestamp) - 1
    if idx < 0 or idx >= len(gnss_data):
        continue

    pos = gnss_data[idx, 1:4]  # 世界坐标系的 (x, y, z)
    rot = heading_data[timestamp]  # 旋转矩阵 R_wc

    # **转换点云**
    P_world = (rot @ points.T).T + pos  # 应用旋转 + 平移
    lidar_transformed.append(P_world)

# **可视化**
gnss_pcd = o3d.geometry.PointCloud()
gnss_pcd.points = o3d.utility.Vector3dVector(gnss_data[:, 1:4])
gnss_pcd.paint_uniform_color([1, 0, 0])  # 轨迹红色

if len(lidar_transformed) > 0:
    lidar_pcd = o3d.geometry.PointCloud()
    all_points = np.vstack(lidar_transformed)

    # 限制点云大小
    if all_points.shape[0] > 500000:
        idx = np.random.choice(all_points.shape[0], size=500000, replace=False)
        all_points = all_points[idx]

    lidar_pcd.points = o3d.utility.Vector3dVector(all_points)
    lidar_pcd.paint_uniform_color([0, 1, 0])  # 绿色表示点云

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(gnss_pcd)
    vis.add_geometry(lidar_pcd)
    vis.run()
    vis.destroy_window()
else:
    print("⚠️ 没有 LiDAR 点云数据")
    o3d.visualization.draw_geometries([gnss_pcd])
