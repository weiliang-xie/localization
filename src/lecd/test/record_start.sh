#!/bin/bash

echo "Starting main program..."
# 启动主程序（catkin build lecd）并获取 PID
echo "Starting catkin build for lecd..."
../../../devel/lib/lecd/lecd_test  &
BUILD_PID=$!
echo "Main program started with PID $BUILD_PID."

INTERVAL=1  # 每秒采样一次
DURATION=3600  # 采样持续60秒

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建日志文件
CPU_LOG="$(pwd)/../../../data_pre/log/localization-log/cpu_usage_$TIMESTAMP.log"
MEMORY_LOG="$(pwd)/../../../data_pre/log/localization-log/memory_usage_$TIMESTAMP.log"

echo "Starting SAR CPU monitoring..."
sar -u $INTERVAL $DURATION > $CPU_LOG &
CPU_PID=$!
echo "SAR CPU monitoring started with PID $CPU_PID."

echo "Starting SAR memory monitoring..."
sar -r $INTERVAL $DURATION > $MEMORY_LOG &
MEMORY_PID=$!
echo "SAR memory monitoring started with PID $MEMORY_PID."


# 等待主程序执行完成
wait $BUILD_PID

# 停止数据记录：程序结束时
kill $CPU_PID
kill $MEMORY_PID

echo "CPU usage and memory usage data have been recorded."
