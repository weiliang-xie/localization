#!/usr/bin/env python3
import os
import re
import time
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient


def extract_number(filename):
    """从文件名中提取数字"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def get_sorted_files(folder_path):
    """按文件名中的数字排序文件"""
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(files, key=extract_number)
    return sorted_files


def send_file_via_scp(ssh_host, ssh_user, ssh_password, remote_folder, file_path):
    """通过SCP发送文件到局域网设备"""
    try:
        # 创建SSH客户端并连接
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

        # 创建SCP客户端
        with SCPClient(ssh.get_transport()) as scp:
            # 上传文件到目标文件夹
            scp.put(file_path, remote_path=remote_folder)

        print(f"文件发送成功: {file_path}")
        ssh.close()
    except Exception as e:
        print(f"发送文件失败: {file_path}, 错误: {e}")


def process_files_with_delay():
    """按顺序发送文件，并在每次发送之间添加延迟"""
    # 配置部分
    folder_path = "./00/velodyne"  # 本地文件夹路径
    ssh_host = "192.168.0.31"  # 目标设备IP地址
    ssh_user = "forlinx"  # SSH用户名
    ssh_password = "forlinx"  # SSH密码
    remote_folder = "/home/forlinx/compress/data"  # 远程目标文件夹
    delay_between_files = 0.1  # 每个文件发送之间的延迟时间（秒）

    # 获取排序后的文件列表
    sorted_files = get_sorted_files(folder_path)
    print(f"按文件名中的数字排序后的文件列表: {sorted_files}")

    # 按顺序发送文件
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        send_file_via_scp(ssh_host, ssh_user, ssh_password, remote_folder, file_path)

        # 每个文件之间添加延迟
        print(f"等待 {delay_between_files} 秒后发送下一个文件...")
        time.sleep(delay_between_files)


def main():
    process_files_with_delay()


if __name__ == "__main__":
    main()

