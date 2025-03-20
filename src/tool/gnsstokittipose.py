import numpy as np
import pandas as pd
import math
import rosbag
from tqdm import tqdm

#用于将gps数据转换成kitti pose的形式

# -------------------------------------------
# 加载 GPS+Heading 数据
# -------------------------------------------
gps_df = pd.read_csv('/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_data.csv')

# 经纬度转米比例（近似）
LAT_TO_M = 111000
lon_ref_rad = math.radians(gps_df['latitude'][0])
LON_TO_M = 111000 * math.cos(lon_ref_rad)

# 初始 GPS 点
lat_ref = gps_df['latitude'][0]
lon_ref = gps_df['longitude'][0]
heading_ref = gps_df['heading'][0]
timestamp_ref = gps_df['timestamp'][0]

# 设定初始时间为第一帧的时间戳
t0 = timestamp_ref

# -------------------------------------------
# 时间同步: 找到最接近的 GPS 时间戳
# -------------------------------------------
gps_timestamps = gps_df['timestamp'].values

def find_nearest_gps_index(lidar_time, gps_times):
    """
    在 GPS 时间序列中找到最接近 lidar_time 的索引
    """
    idx = np.abs(gps_times - lidar_time).argmin()
    return idx

# -------------------------------------------
# 构建 Pose 文件
# -------------------------------------------
pose_list = []

# 遍历 GPS 时间戳
for gps_idx in tqdm(range(len(gps_timestamps)), desc="处理位姿"):
    gps_row = gps_df.iloc[gps_idx]

    # 计算相对时间戳（以 GPS 首帧为基准）
    relative_time = gps_timestamps[gps_idx] - t0

    # 计算相对平移
    d_lat = (gps_row['latitude'] - lat_ref) * LAT_TO_M
    d_lon = (gps_row['longitude'] - lon_ref) * LON_TO_M
    d_z = 0  # z轴平移设为0

    # 相对朝向差（heading）
    heading = gps_row['heading']
    d_heading = math.radians(heading - heading_ref)

    # 构建旋转矩阵（仅绕 z 轴旋转）
    cos_h = math.cos(d_heading)
    sin_h = math.sin(d_heading)
    R = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1]
    ])

    # 构建 3x4 变换矩阵
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:, 3] = [d_lon, d_lat, d_z]  # x: 经度方向，y: 纬度方向

    # 转为一维数组（时间戳 + 12个数）
    T_row = [relative_time] + T.flatten().tolist()
    pose_list.append(T_row)

# -------------------------------------------
# 保存 Pose 文件
# -------------------------------------------
pose_array = np.array(pose_list)
np.savetxt('/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gt_pose_xuda.txt', pose_array, fmt='%.6f', delimiter=' ')
print("同步后的位姿文件已生成：gt_pose_xuda.txt")
