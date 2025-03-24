import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import pandas as pd


# 配置参数
bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-03-15-18-30-36.bag"  # 你的.bag文件路径
lidar_topic = "/lidar_points"  # LiDAR数据话题
output_folder = "/home/jtcx/remote_control/code/localization/data_pre/lidar-data/xuda-less-0.5"  # 输出bin文件的文件夹
gps_df = pd.read_csv('/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_data.csv') #gps时间戳和定位信息


# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 提取 GPS 时间戳
gps_timestamps = gps_df['timestamp'].to_numpy()

# 读取bag文件
lidar_data = []
lidar_timestamps = []

with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[lidar_topic]):
        timestamp = t.to_sec()  # 提取时间戳
        lidar_timestamps.append(timestamp)

        # 解析点云数据
        point_cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
        lidar_data.append(point_cloud)

# 转换为 NumPy 数组
lidar_timestamps = np.array(lidar_timestamps)
lidar_data = np.array(lidar_data, dtype=object)


# —————————————————————————— GPS 与 LiDAR 时间戳对齐 ——————————————————————————
def get_lidar_indices(gps_timestamps, lidar_timestamps):
    """
    使用 np.searchsorted 对齐 GPS 和 LiDAR 数据时间戳
    返回与 GPS 时间戳最接近的 LiDAR 数据索引
    """
    lidar_indices = np.searchsorted(lidar_timestamps, gps_timestamps)

    # 确保索引不会越界
    lidar_indices[lidar_indices >= len(lidar_timestamps)] = len(lidar_timestamps) - 1

    # 检查时间戳是否在合理误差范围内
    max_time_diff = 0.9  # 允许的最大时间差 (秒)
    valid_mask = np.abs(lidar_timestamps[lidar_indices] - gps_timestamps) <= max_time_diff

    if not np.all(valid_mask):
        print(f"⚠️ 警告：发现 {np.sum(~valid_mask)} 个 GPS 时间戳未找到有效的 LiDAR 数据匹配。")

    return lidar_indices[valid_mask]

# 获取与 GPS 时间戳对齐的 LiDAR 数据索引
lidar_indices = get_lidar_indices(gps_timestamps, lidar_timestamps)

# —————————————————————————— 保存对齐的 LiDAR 数据 ——————————————————————————
for idx, frame_idx in zip(lidar_indices, range(len(lidar_indices))):
    bin_filename = os.path.join(output_folder, f"{frame_idx:06d}.bin")

    # 保存点云数据
    lidar_data[idx].astype(np.float32).tofile(bin_filename)

    print(f"✅ Saved {bin_filename}")

print("✅ LiDAR 数据提取并对齐完成！")
