import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2

# 配置参数
bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-03-15-18-30-36.bag"  # 你的.bag文件路径
lidar_topic = "/lidar_points"  # LiDAR数据话题
output_folder = "/home/jtcx/remote_control/code/localization/data_pre/lidar-data/xuda"  # 输出bin文件的文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取bag文件
bag = rosbag.Bag(bag_file, "r")

# 遍历消息
frame_idx = 0
for topic, msg, t in bag.read_messages(topics=[lidar_topic]):
    # 解析点云数据
    point_cloud = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))

    # 构造文件名
    bin_filename = os.path.join(output_folder, f"{frame_idx:06d}.bin")

    # 保存到bin文件
    point_cloud.astype(np.float32).tofile(bin_filename)

    print(f"Saved {bin_filename}")
    frame_idx += 1

# 关闭bag文件
bag.close()

print("LiDAR 数据提取完成！")
