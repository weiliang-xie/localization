import rosbag
import numpy as np
import utm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import pandas as pd

#*合并/novatel718d/pos和/novatel718d/heading话题，组成位姿，引入速度利用卡尔曼滤波优化数据，生成轨迹图和gps数据文件

# 读取的 rosbag 文件路径
bag_file = "/home/jtcx/data_set/self/xuda/mapping_2025-03-15-18-30-36.bag"  #xuda
# bag_file = "/home/jtcx/data_set/self/factory/mapping_2025-03-18-10-53-21.bag"  #factory

# 存储 GPS 数据
gnss_data = []
# 存储 GPS 数据
gps_timestamps = []
utm_x_list = []
utm_y_list = []
speed_list = []  # 速度 (m/s)
heading_list = []

speed_timestamps = []  # 速度时间戳
speed_data = []  # 速度 (m/s)

heading_timestamps = []  # 航向角时间戳
heading_data = []  # 航向角 (Yaw)


def quaternion_to_yaw(qx, qy, qz, qw):
    """将四元数转换为偏航角（Yaw）"""
    try:
        rotation = R.from_quat([qx, qy, qz, qw])
        euler = rotation.as_euler('xyz', degrees=True)  # 转换为欧拉角 (°)
        return euler[2]  # 返回偏航角（Yaw）
    except ValueError:
        return None  # 处理异常情况，返回None

def convert_gps_to_utm(lat, lon, status):
    """检查纬度范围并转换为 UTM 坐标"""
    if not (-80.0 <= lat <= 84.0):
        print(f"⚠️ 无效纬度: {lat} 超出范围！跳过转换， 定位状态：{status}")
        #加入插值
        
        return None, None  # 过滤错误数据
    return utm.from_latlon(lat, lon)[:2]  # 只返回 UTM x, y

# **插值填补丢失的航向角**
def interpolate_missing_values(time_series, values):
    """插值填补丢失的数据"""

    # **确保输入数据是 NumPy 数组**
    time_series = np.array(time_series, dtype=float)
    values = np.array(values, dtype=float)

    # **检查并找到有效数据索引**
    # **将 None 转换为 NaN**
    values = np.array([float('nan') if v is None else v for v in values], dtype=float)
    valid_indices = np.where(~np.isnan(values))[0]  # 只取非 NaN 索引
    if len(valid_indices) < 2:  # 需要至少 2 个有效数据点进行插值
        print("⚠️ 有效航向角数据不足，无法进行插值！")
        return values  # 直接返回原数据（包含 NaN）

    # **执行线性插值**
    interpolator = interp1d(time_series[valid_indices], values[valid_indices], kind='linear', fill_value="extrapolate")
    print("完成航向角数据插值！")

    # **返回填补后的数据**
    return interpolator(time_series)

#卡尔曼滤波
def apply_kalman_filter(x_values, y_values):
    """使用卡尔曼滤波平滑 GPS 轨迹"""
    # 检查输入数据
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("输入数据不能为空")
    if len(x_values) != len(y_values):
        raise ValueError("x_values 和 y_values 的长度必须相同")
    if np.any(np.isnan(x_values)) or np.any(np.isnan(y_values)):
        raise ValueError("输入数据包含 NaN 值")
    if np.any(np.isinf(x_values)) or np.any(np.isinf(y_values)):
        raise ValueError("输入数据包含 inf 值")
    if len(x_values) < 2:
        raise ValueError("输入数据点太少，至少需要 2 个点")

    # 配置卡尔曼滤波
    kf = KalmanFilter(
        initial_state_mean=[x_values[0], y_values[0]],
        n_dim_obs=2,
        transition_matrices=np.eye(2),  # 状态转移矩阵
        observation_matrices=np.eye(2),  # 观测矩阵
        initial_state_covariance=np.eye(2) * 1,  # 初始状态协方差
        transition_covariance=np.eye(2) * 0.01,  # 过程噪声
        observation_covariance=np.eye(2) * 0.1  # 观测噪声
    )

    # 应用卡尔曼滤波
    try:
        smoothed_state_means, _ = kf.smooth(np.column_stack((x_values, y_values)))
    except Exception as e:
        raise RuntimeError(f"卡尔曼滤波失败: {e}")

    # 检查结果有效性
    if np.any(np.isnan(smoothed_state_means)):
        raise RuntimeError("卡尔曼滤波返回了 NaN 值")

    return smoothed_state_means[:, 0], smoothed_state_means[:, 1]

def apply_kalman_filter_with_velocity(timestamps, x_values, y_values, speed_values, headings):
    """使用卡尔曼滤波优化GPS轨迹，引入速度信息"""

    Δt = np.mean(np.diff(timestamps))  # 计算平均时间间隔

    # **状态转移矩阵 F**
    F = np.array([[1, 0, Δt, 0], 
                  [0, 1, 0, Δt], 
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1]])

    # **观测矩阵 H**（GPS 只能测量 `x, y`）
    H = np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0]])

    # **观测噪声协方差 R**（GPS 测量误差）
    R = np.eye(2) * 5  # GPS 误差（单位：米）

    # **状态噪声协方差 Q**（运动预测误差）
    Q = np.eye(4) * 0.1  # 过程噪声

    # **初始状态 [x, y, v_x, v_y]**
    v_x_init = speed_values[0] * np.cos(np.radians(headings[0]))  # 速度 x 分量
    v_y_init = speed_values[0] * np.sin(np.radians(headings[0]))  # 速度 y 分量
    initial_state = [x_values[0], y_values[0], v_x_init, v_y_init]

    kf = KalmanFilter(initial_state_mean=initial_state,
                      transition_matrices=F,
                      observation_matrices=H,
                      observation_covariance=R,
                      transition_covariance=Q)

    # **执行滤波**
    measurements = np.column_stack((x_values, y_values))  # GPS 位置观测值
    smoothed_state_means, _ = kf.smooth(measurements)

    # 提取平滑后的 `x, y`
    x_smooth = smoothed_state_means[:, 0]
    y_smooth = smoothed_state_means[:, 1]

    return x_smooth, y_smooth


#————————————————————————————————————提取数据————————————————————————————————————#
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        timestamp = t.to_sec()  # 直接使用 ROS Bag 记录的时间戳
        if topic == "/novatel718d/pos":
            print(f"📌 处理 GPS 数据: lat={msg.latitude}, lon={msg.longitude}")

            # 过滤无效数据
            if msg.latitude == 0.0 and msg.longitude == 0.0:
                print("⚠️ 发现无效 GPS 数据 (0,0)，跳过！")
                continue

            # 坐标转换
            utm_x, utm_y = convert_gps_to_utm(msg.latitude, msg.longitude, msg.status)

            # if utm_x is not None and utm_y is not None:
            gps_timestamps.append(timestamp)  # 时间戳
            utm_x_list.append(utm_x)
            utm_y_list.append(utm_y)

        # 处理 速度 数据
        elif topic == "/chassis":
            speed_timestamps.append(timestamp)  # 这里手动添加时间戳
            speed_data.append(msg.mcu.speed_mps)  # 直接存储速度 (m/s)

        # 处理 航向角 (Quaternion) 数据
        elif topic == "/novatel718d/heading":
            yaw_angle = quaternion_to_yaw(msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w)
            heading_timestamps.append(timestamp)
            heading_data.append(yaw_angle)


# 转换为 NumPy 数组
gps_timestamps = np.array(gps_timestamps)
utm_x_list = np.array(utm_x_list)
utm_y_list = np.array(utm_y_list)

# **确保 heading_timestamps 和 heading_data 是 NumPy 数组**
heading_timestamps = np.array(heading_timestamps, dtype=float)
heading_data = np.array(heading_data, dtype=float)

#————————————————————————————————————填充对齐数据————————————————————————————————————#
# **检查航向角数据是否为空**
if len(heading_timestamps) < 2 or len(heading_data) < 2:
    print("⚠️ 航向角数据不足，无法进行插值！")
    heading_data_filled = np.full_like(heading_timestamps, np.nan)  # 用 NaN 填充
else:
    heading_data_filled = interpolate_missing_values(heading_timestamps, np.array(heading_data, dtype=float))


speed_interp = interp1d(speed_timestamps, speed_data, kind='linear', fill_value="extrapolate")
heading_interp = interp1d(heading_timestamps, heading_data_filled, kind='linear', fill_value="extrapolate")

gps_speeds = speed_interp(gps_timestamps)  # 计算GPS时间点对应的速度
gps_headings = heading_interp(gps_timestamps)  # 计算GPS时间点对应的航向角

speed_list = np.array(gps_speeds)
heading_list = np.array(gps_headings)
            
# 查找缺失数据
# **将 None 转换为 NaN**
utm_x_list = np.array([float('nan') if v is None else v for v in utm_x_list], dtype=float)
utm_y_list = np.array([float('nan') if v is None else v for v in utm_y_list], dtype=float)

#去除数据开始的nan值
first_valid_index = np.argmax(~np.isnan(utm_x_list))  # 找到第一个非NaN元素的位置
# 截取有效数据（删除开头的连续NaN）
utm_x_list = utm_x_list[first_valid_index:]
utm_y_list = utm_y_list[first_valid_index:]
gps_timestamps = gps_timestamps[first_valid_index:]
speed_list = speed_list[first_valid_index:]
heading_list = heading_list[first_valid_index:]

nan_indices = np.where(np.isnan(utm_x_list) | np.isnan(utm_y_list))[0]

print(f"缺失的数据点数量为: {nan_indices.size}")

# 利用速度推算插值填补缺失的GPS点
for idx in nan_indices:
    if 0 < idx < len(utm_x_list):
        dt = gps_timestamps[idx] - gps_timestamps[idx - 1]  # 计算时间间隔
        if dt > 0:
            heading_rad = np.radians(heading_list[idx - 1])  # 转换为弧度
            utm_x_list[idx] = utm_x_list[idx - 1] + speed_list[idx - 1] * dt * np.sin(heading_rad)
            utm_y_list[idx] = utm_y_list[idx - 1] + speed_list[idx - 1] * dt * np.cos(heading_rad)
            print(f"🔄 插值填补点 {idx}: X={utm_x_list[idx]:.3f}, Y={utm_y_list[idx]:.3f}")
        else:
            utm_x_list = np.delete(utm_x_list, idx)
            utm_y_list = np.delete(utm_y_list, idx)
            gps_timestamps = np.delete(gps_timestamps, idx)
            speed_list = np.delete(speed_list, idx)
            heading_list = np.delete(heading_list, idx)
            
    # else:
        # utm_x_list = np.delete(utm_x_list, idx)
        # utm_y_list = np.delete(utm_y_list, idx)
        # gps_timestamps = np.delete(gps_timestamps, idx)
        # speed_list = np.delete(speed_list, idx)
        # heading_list = np.delete(heading_list, idx)


#判断是否仍有nan值
# if np.any(np.isnan(utm_x_list)) or np.any(np.isnan(utm_y_list)):
#     raise ValueError("插值过后的数据仍包含 NaN 值")

# 判断并打印 NaN 的索引
nan_indices_x = np.where(np.isnan(utm_x_list))[0]  # 找到 X 中 NaN 的索引
nan_indices_y = np.where(np.isnan(utm_y_list))[0]  # 找到 Y 中 NaN 的索引

if nan_indices_x.size > 0 or nan_indices_y.size > 0:
    error_message = "插值过后的数据仍包含 NaN 值\n"
    error_message += f"NaN 索引: {nan_indices_x.tolist()}\n"
    raise ValueError(error_message)
else:
    print("✅ 数据无 NaN 值")

print(f"插值后的数据点数量为: {utm_x_list.size}")


#————————————————————————————————————去除异常点————————————————————————————————————#
# 获取排序索引
sorted_indices = np.argsort(gps_timestamps)

# 使用排序索引对所有数组排序
gps_timestamps = gps_timestamps[sorted_indices]
utm_x_list = utm_x_list[sorted_indices]
utm_y_list = utm_y_list[sorted_indices]
speed_list = speed_list[sorted_indices]
heading_list = heading_list[sorted_indices]

abnormal_indices = []
repeat_indices = []
while 1:
    #去除异常点
    # 设置合理的速度（单位：m/s）
    MAX_VELOCITY = 2  # 可以根据实际情况调整
    MIN_VELOCITY = 0.3  # 可以根据实际情况调整
    # 计算相邻点之间的位移和时间间隔
    dx = np.diff(utm_x_list)  # X 方向的位移
    dy = np.diff(utm_y_list)  # Y 方向的位移
    dt = np.diff(gps_timestamps)  # 时间间隔
    # 计算瞬时速度
    distance = np.sqrt(dx**2 + dy**2)  # 计算欧几里得距离
    velocity = distance / dt  # 计算速度
    # 找出速度异常的点（超出 MAX_VELOCITY 的索引）
    abnormal_indices = np.where(velocity > MAX_VELOCITY)[0] + 1  # +1 因为 diff() 计算的是前后点的差值
    repeat_indices = np.where(velocity < MIN_VELOCITY)[0] + 1  # +1 因为 diff() 计算的是前后点的差值
    # 合并并去重排序
    combined_indices = np.union1d(abnormal_indices, repeat_indices)

    print(f"🚨 发现 {len(abnormal_indices)} 个异常点（速度过大）：{abnormal_indices}")
    print(f"🚨 发现 {len(repeat_indices)} 个重复点（速度过小）：{repeat_indices}")
    # **去除异常点**
    valid_indices = np.setdiff1d(np.arange(len(utm_x_list)), combined_indices)  # 仅保留正常数据

    utm_x_list = utm_x_list[valid_indices]
    utm_y_list = utm_y_list[valid_indices]
    gps_timestamps = gps_timestamps[valid_indices]
    speed_list = speed_list[valid_indices]
    heading_list = heading_list[valid_indices]

    print(f"去除异常点后的数据点数量为: {utm_x_list.size}")
    if len(abnormal_indices) == 0:
        break

#————————————————————————————————————去除偏航点 xuda————————————————————————————————————#
# 异常点的坐标范围 (根据红框的坐标估计)
x_min, x_max = 802244, 802256
y_min, y_max = 2494440, 2494454

# 找出不在异常范围内的索引
abnormal_indices_ = np.where((utm_x_list >= x_min) & (utm_x_list <= x_max) &
                             (utm_y_list >= y_min) & (utm_y_list <= y_max))[0]

valid_indices_ = np.setdiff1d(np.arange(len(utm_x_list)), abnormal_indices_)  # 仅保留正常数据


print(f"🚨 发现 {len(abnormal_indices_)} 个异常点：{abnormal_indices_}")


# 根据索引筛选出正常数据
utm_x_list = utm_x_list[valid_indices_]
utm_y_list = utm_y_list[valid_indices_]
gps_timestamps = gps_timestamps[valid_indices_]
speed_list = speed_list[valid_indices_]
heading_list = heading_list[valid_indices_]

print(f"去除异常点后的数据点数量为: {utm_x_list.size}")


#————————————————————————————————————卡尔曼滤波————————————————————————————————————#

utm_x_list_kalman, utm_y_list_kalman = apply_kalman_filter_with_velocity(gps_timestamps, utm_x_list, utm_y_list, speed_list, heading_list)
# utm_x_list_kalman, utm_y_list_kalman = utm_x_list, utm_y_list

# print(f"卡尔曼滤波后的数据点数量为: {utm_x_list_kalman.size}")


#————————————————————————————————————数据保存————————————————————————————————————#
# **将 UTM 坐标转换回经纬度**
# **选取轨迹的第一个点进行 UTM 转换**
if len(utm_x_list_kalman) > 0:
    example_lat, example_lon = 22.5305, 113.9393  # 替换为你的 GPS 数据
    _, _, utm_zone, _ = utm.from_latlon(example_lat, example_lon)  # 提取正确的 `zone_number`
    print(f"✅ 使用 UTM Zone: {utm_zone}")
else:
    raise ValueError("❌ UTM 数据为空！")
# **转换 UTM 坐标到 经纬度**
lat_lon_list = [utm.to_latlon(x, y, utm_zone, northern=True) for x, y in zip(utm_x_list_kalman, utm_y_list_kalman)]

# **构建 CSV 数据**
csv_data = {
    "timestamp": gps_timestamps,
    "latitude": [lat for lat, lon in lat_lon_list],  # 提取纬度
    "longitude": [lon for lat, lon in lat_lon_list],  # 提取经度
    "heading": heading_list,
    "speed": speed_list
}

# **转换为 DataFrame**
df = pd.DataFrame(csv_data)

# **保存 CSV 文件**
csv_filename = "/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_data.csv"  #xuda
# csv_filename = "/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_data_only.csv"  #xuda
# csv_filename = "/home/jtcx/remote_control/code/localization/data_pre/gtpose/factory/gps_data.csv"  #factory
df.to_csv(csv_filename, index=False)

print(f"✅ 轨迹数据已保存为 CSV 文件: {csv_filename}")


#————————————————————————————————————绘制图表————————————————————————————————————#

# 绘制原始轨迹 vs. 优化轨迹
plt.figure(figsize=(10, 6))
# **绘制卡尔曼轨迹**
plt.plot(utm_x_list_kalman, utm_y_list_kalman, marker=".", linestyle="-", label="Kalman Path", color="green")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Optimized GNSS Trajectory Visualization")
plt.legend()
plt.grid()

# # 🔍 添加放大图
# ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')

# # 在放大图中绘制相同轨迹并缩放指定区域
# ax_inset.plot(utm_x_list_kalman, utm_y_list_kalman, marker=".", linestyle="-", color="green")

# # 设置放大区域的坐标范围（根据数据调整）
# ax_inset.set_xlim(802320, 802340)
# ax_inset.set_ylim(2494442, 2494462)
# # 🔹 去除放大图的坐标轴
# ax_inset.axis("off")
# # 🔹 在主图中标注放大区域并添加指引线
# mark_inset(plt.gca(), ax_inset, loc1=2, loc2=4, fc="none", ec="red", lw=1.5)

plt.savefig("/home/jtcx/ICRA/exper_data_1.0/thesis/pdf/gps_xuda_path.pdf", format="pdf", bbox_inches="tight")    #xuda
# plt.savefig("/home/jtcx/ICRA/exper_data_1.0/thesis/pdf/gps_xuda_path-kfbefore.pdf", format="pdf", bbox_inches="tight")    #xuda
# plt.savefig("/home/jtcx/ICRA/exper_data_1.0/thesis/pdf/gps_factory_path.pdf", format="pdf", bbox_inches="tight")    #factory
plt.show()