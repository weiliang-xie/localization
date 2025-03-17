import rosbag
import numpy as np
import utm
import matplotlib.pyplot as plt

# bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-02-13-16-42-06.bag"
# bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-02-13-16-47-55.bag"
# bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-02-13-16-54-32.bag"
# bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-03-15-18-03-59.bag"
bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-03-15-18-30-36.bag"
gnss_data = []
heading_data = []

def convert_gps_to_utm(lat, lon, status):
    """检查纬度范围并转换 UTM"""
    if not (-80.0 <= lat <= 84.0):
        print(f"⚠️ 无效纬度: {lat} 超出范围！跳过转换， 定位状态：{status}")
        return None, None  # 过滤错误数据
    return utm.from_latlon(lat, lon)[:2]  # 只返回 UTM x, y

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == "/novatel718d/pos":
            print(f"📌 处理 GPS 数据: lat={msg.latitude}, lon={msg.longitude}")

            # 过滤错误数据
            if msg.latitude == 0.0 and msg.longitude == 0.0:
                print("⚠️ 发现无效 GPS 数据 (0,0)，跳过！")
                continue

            utm_x, utm_y = convert_gps_to_utm(msg.latitude, msg.longitude, msg.status)
            if utm_x is not None and utm_y is not None:
                gnss_data.append((utm_x, utm_y))

# 转换数据
gnss_data = np.array(gnss_data)
x = gnss_data[:, 0] - gnss_data[0, 0]
y = gnss_data[:, 1] - gnss_data[0, 1]

# 绘制轨迹
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker=".", linestyle="-", label="GNSS Path")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("GNSS Trajectory Visualization")
plt.legend()
plt.grid()
plt.show()
