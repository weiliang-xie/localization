import matplotlib.pyplot as plt
from datetime import datetime

#02.13测试结果：cpu_usage_20250213_103743（cc） cpu_usage_20250212_225005（lecd）

# 固定日志文件路径
log_file_1 = '/home/jtcx/remote_control/code/localization/data_pre/log/compress-log/cpu_usage_20250213_103743.log'  # 程序1的日志文件
log_file_2 = '/home/jtcx/remote_control/code/localization/data_pre/log/compress-log/cpu_usage_20250212_225005.log'  # 程序2的日志文件
# log_file_1 = '/home/jtcx/remote_control/code/localization/data_pre/log/localization-log/cpu_usage_20250212_211622.log'  # 程序1的日志文件
# log_file_2 = '/home/jtcx/remote_control/code/localization/data_pre/log/localization-log/cpu_usage_20250212_214707.log'  # 程序2的日志文件

log_mem_file_1 = '/home/jtcx/remote_control/code/localization/data_pre/log/compress-log/memory_usage_20250213_103743.log'  # 程序1的内存日志文件
log_mem_file_2 = '/home/jtcx/remote_control/code/localization/data_pre/log/compress-log/memory_usage_20250212_225005.log'  # 程序2的内存日志文件

# 输入两个程序的时间戳范围
start_time_1 = '10:37:48'  # 程序1的起始时间戳
end_time_1 = '11:01:00'    # 程序1的终止时间戳  22
start_time_2 = '22:50:10'  # 程序2的起始时间戳
end_time_2 = '23:13:10'    # 程序2的终止时间戳  18

# 将时间戳转换为 datetime 对象
start_time_1 = datetime.strptime(start_time_1, '%H:%M:%S')
end_time_1 = datetime.strptime(end_time_1, '%H:%M:%S')
start_time_2 = datetime.strptime(start_time_2, '%H:%M:%S')
end_time_2 = datetime.strptime(end_time_2, '%H:%M:%S')

# 读取程序1的日志文件并提取数据
user_data_1 = []
idle_data_1 = []
cpu_line_numbers_1 = []  # 存储程序1的CPU日志的行号
with open(log_file_1, 'r') as file:
    for line_number, line in enumerate(file, 1):  # 使用line_number跟踪行号
        parts = line.split()

        # 跳过不包含时间戳的行（如头部信息）
        if len(parts) < 7 or not parts[0].count(':') == 2:
            continue
        
        try:
            # 提取时间戳（HH:MM:SS）和CPU使用情况
            timestamp = datetime.strptime(parts[0], '%H:%M:%S')
        except ValueError:
            # 如果无法解析时间戳，跳过该行
            continue
        
        # 提取程序1的数据
        if start_time_1 <= timestamp <= end_time_1:
            user_1 = float(parts[2])  # %user
            idle_1 = float(parts[7])  # %idle
            if idle_1 < 60:
                user_data_1.append(user_1)
                idle_data_1.append(idle_1)
                cpu_line_numbers_1.append(line_number)  # 记录该行的行号

# 读取程序2的日志文件并提取数据
user_data_2 = []
idle_data_2 = []
cpu_line_numbers_2 = []  # 存储程序2的CPU日志的行号
with open(log_file_2, 'r') as file:
    for line_number, line in enumerate(file, 1):  # 使用line_number跟踪行号
        parts = line.split()
        
        # 跳过不包含时间戳的行（如头部信息）
        if len(parts) < 7 or not parts[0].count(':') == 2:
            continue
        
        try:
            # 提取时间戳（HH:MM:SS）和CPU使用情况
            timestamp = datetime.strptime(parts[0], '%H:%M:%S')
        except ValueError:
            # 如果无法解析时间戳，跳过该行
            continue
        
        # 提取程序2的数据
        if start_time_2 <= timestamp <= end_time_2:
            user_2 = float(parts[2])  # %user
            idle_2 = float(parts[7])  # %idle
            if idle_2 < 60:
                user_data_2.append(user_2)
                idle_data_2.append(idle_2)
                cpu_line_numbers_2.append(line_number)  # 记录该行的行号

# 读取程序1的内存日志文件并提取数据
memused_data_1 = []
kbbuffers_data_1 = []
with open(log_mem_file_1, 'r') as file:
    for line_number, line in enumerate(file, 1):  # 使用line_number跟踪行号
        parts = line.split()
        
        # 跳过不包含时间戳的行（如头部信息）
        if len(parts) < 7 or not parts[0].count(':') == 2:
            continue
        
        try:
            # 提取时间戳（HH:MM:SS）和内存使用情况
            timestamp = datetime.strptime(parts[0], '%H:%M:%S')
        except ValueError:
            # 如果无法解析时间戳，跳过该行
            continue
        
        # 提取程序1的内存数据
        if line_number in cpu_line_numbers_1:
            memused_1 = float(parts[4])  # %memused
            kbbuffers_1 = float(parts[5])  # kbbuffers
            memused_data_1.append(memused_1)
            kbbuffers_data_1.append(kbbuffers_1)

# 读取程序2的内存日志文件并提取数据
memused_data_2 = []
kbbuffers_data_2 = []
with open(log_mem_file_2, 'r') as file:
    for line_number, line in enumerate(file, 1):  # 使用line_number跟踪行号
        parts = line.split()
        
        # 跳过不包含时间戳的行（如头部信息）
        if len(parts) < 7 or not parts[0].count(':') == 2:
            continue
        
        try:
            # 提取时间戳（HH:MM:SS）和内存使用情况
            timestamp = datetime.strptime(parts[0], '%H:%M:%S')
        except ValueError:
            # 如果无法解析时间戳，跳过该行
            continue
        
        # 提取程序2的内存数据
        if line_number in cpu_line_numbers_2:
            memused_2 = float(parts[4])  # %memused
            kbbuffers_2 = float(parts[5])  # kbbuffers
            memused_data_2.append(memused_2)
            kbbuffers_data_2.append(kbbuffers_2)


# 计算程序1和程序2的 %user 和 %idle 的平均值
def calculate_average(data):
    return sum(data) / len(data) if data else 0

average_user_1 = calculate_average(user_data_1)
average_idle_1 = calculate_average(idle_data_1)
average_user_2 = calculate_average(user_data_2)
average_idle_2 = calculate_average(idle_data_2)

average_memused_1 = calculate_average(memused_data_1)
average_kbbuffers_1 = calculate_average(kbbuffers_data_1)
average_memused_2 = calculate_average(memused_data_2)
average_kbbuffers_2 = calculate_average(kbbuffers_data_2)

# 打印平均值
print(f"Program 1 - Average %user: {average_user_1:.2f}%")
print(f"Program 1 - Average %idle: {average_idle_1:.2f}%")
print(f"Program 2 - Average %user: {average_user_2:.2f}%")
print(f"Program 2 - Average %idle: {average_idle_2:.2f}%")

print(f"Program 1 - Average %memused: {average_memused_1:.2f}%")
print(f"Program 1 - Average kbbuffers: {average_kbbuffers_1:.2f} KB")
print(f"Program 2 - Average %memused: {average_memused_2:.2f}%")
print(f"Program 2 - Average kbbuffers: {average_kbbuffers_2:.2f} KB")


# 创建子图
fig, axs = plt.subplots(4, 1, figsize=(12, 20))

# 提取idle_data_1的序号列表
seq_1 = list(range(1, len(user_data_1) + 1))  # 创建一个从1开始的序号列表
seq_2 = list(range(1, len(user_data_2) + 1))  # 创建一个从1开始的序号列表

# 绘制程序1和程序2的 %user
axs[0].plot(seq_1, user_data_1, label='Program 1 - %user', marker=None, color='#8DA0CB')
axs[0].plot(seq_2, user_data_2, label='Program 2 - %user', marker=None, color='#66C2A5')
axs[0].set_title('CPU Usage - %user')
axs[0].set_xlabel(' ')
axs[0].set_ylabel('Percentage')
axs[0].set_xticks([])  # 隐藏横坐标
axs[0].legend()
axs[0].grid(True)

# 绘制程序1和程序2的 %idle
axs[1].plot(seq_1, idle_data_1, label='Program 1 - %idle', marker=None, color='#8DA0CB')
axs[1].plot(seq_2, idle_data_2, label='Program 2 - %idle', marker=None, color='#66C2A5')
axs[1].set_title('CPU Usage - %idle')
axs[1].set_xlabel(' ')
axs[1].set_ylabel('Percentage')
axs[1].set_xticks([])  # 隐藏横坐标
axs[1].legend()
axs[1].grid(True)

# 绘制程序1和程序2的 %memused
axs[2].plot(seq_1, memused_data_1, label='Program 1 - %memused', marker=None, color='#8DA0CB')
axs[2].plot(seq_2, memused_data_2, label='Program 2 - %memused', marker=None, color='#66C2A5')
axs[2].set_title('Memory Usage - %memused')
axs[2].set_xlabel(' ')
axs[2].set_ylabel('Percentage')
axs[2].set_xticks([])  # 隐藏横坐标
axs[2].legend()
axs[2].grid(True)

# 绘制程序1和程序2的 kbbuffers
axs[3].plot(seq_1, kbbuffers_data_1, label='Program 1 - kbbuffers', marker=None, color='#8DA0CB')
axs[3].plot(seq_2, kbbuffers_data_2, label='Program 2 - kbbuffers', marker=None, color='#66C2A5')
axs[3].set_title('Memory Usage - kbbuffers')
axs[3].set_xlabel(' ')
axs[3].set_ylabel('KB')
axs[3].set_xticks([])  # 隐藏横坐标
axs[3].legend()
axs[3].grid(True)

# # 自动调整布局，增加子图之间的间距
# plt.tight_layout(pad=4.0)

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()
