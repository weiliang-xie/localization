#!/usr/bin/env python3
import os
import re
import time
import shutil
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient


def extract_number(filename):
    """从文件名中提取数字"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def get_sorted_files(folder_path_query):
    """按文件名中的数字排序文件"""
    files = os.listdir(folder_path_query)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path_query, f))]
    sorted_files = sorted(files, key=extract_number)
    return sorted_files

def copy_file(file_path):
    """复制文件到新的位置"""
    dir_name, file_name = os.path.split(file_path)
    new_file_name = f"copy_of_{file_name}"
    new_file_path = os.path.join(dir_name, new_file_name)
    shutil.copy(file_path, new_file_path)  # 复制文件
    return new_file_path


def rename_files_in_bulk(folder_path, suffix):
    """批量重命名文件，确保每个文件都添加指定的后缀"""
    files = get_sorted_files(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        # 直接重命名文件并添加后缀（如果还没有添加）
        rename_file_with_suffix(file_path, suffix)


def rename_file_with_suffix(file_path, suffix):
    """检查文件名是否已包含后缀，若未包含则修改文件名并加上后缀"""
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)

    # 如果文件名已经包含后缀，则不修改
    if suffix not in name:
        new_file_name = f"{name}{suffix}{ext}"  # 添加后缀
        new_file_path = os.path.join(dir_name, new_file_name)
        os.rename(file_path, new_file_path)  # 重命名文件
        print(f"文件重命名: {file_name} -> {new_file_name}")
    else:
        print(f"文件 {file_name} 已包含后缀 {suffix}，无需修改")


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

def delete_file(file_path):
    """删除文件"""
    try:
        os.remove(file_path)
        print(f"文件已删除: {file_path}")
    except Exception as e:
        print(f"删除文件失败: {file_path}, 错误: {e}")


def process_files_with_delay():
    """按顺序发送文件，并在每次发送之间添加延迟"""
    # 配置部分
    folder_path_query = "./00/velodyne"  # 本地文件夹路径
    ssh_host = "192.168.0.31"  # 目标设备IP地址
    ssh_user = "forlinx"  # SSH用户名
    ssh_password = "forlinx"  # SSH密码
    remote_folder = "/home/forlinx/compress/data"  # 远程目标文件夹
    delay_between_files = 0.0  # 每个文件发送之间的延迟时间（秒）
    suffix_query = "_query"  # 要添加到文件名的英文后缀

    folder_path_map = "./map"  # 本地文件夹路径
    suffix_map = "_map"  # 要添加到文件名的英文后缀

    # 批量重命名文件，确保每个文件都包含所需的后缀
    rename_files_in_bulk(folder_path_map, suffix_map)
    rename_files_in_bulk(folder_path_query, suffix_query)

    # 获取排序后的文件列表
    sorted_files = get_sorted_files(folder_path_map)
    print(f"按文件名中的数字排序后的文件列表: {sorted_files}")

    # 按顺序发送文件
    for file_name in sorted_files:
        file_path = os.path.join(folder_path_map, file_name)

        # # 复制文件到新位置
        # copied_file_path = copy_file(file_path)
        # print(f"文件已复制: {copied_file_path}")        
        # 修改文件名并加上后缀
        # new_file_path = rename_file_with_suffix(file_path, suffix_map)
        # print(f"文件已重命名: {new_file_path}")

        # 发送修改后的文件
        send_file_via_scp(ssh_host, ssh_user, ssh_password, remote_folder, new_file_path)

        # # 删除已经发送的文件
        # delete_file(new_file_path)

        # 每个文件之间添加延迟
        # print(f"等待 {delay_between_files} 秒后发送下一个文件...")
        time.sleep(delay_between_files)

    # 获取排序后的文件列表
    sorted_files = get_sorted_files(folder_path_query)
    print(f"按文件名中的数字排序后的文件列表: {sorted_files}")

    # 按顺序发送文件
    for file_name in sorted_files:
        file_path = os.path.join(folder_path_query, file_name)
        
        # # 复制文件到新位置
        # copied_file_path = copy_file(file_path)
        # print(f"文件已复制: {copied_file_path}")        
        # 修改文件名并加上后缀
        # new_file_path = rename_file_with_suffix(file_path, suffix_query)
        # print(f"文件已重命名: {new_file_path}")

        # 发送修改后的文件
        send_file_via_scp(ssh_host, ssh_user, ssh_password, remote_folder, file_path)

        # # 删除已经发送的文件
        # delete_file(new_file_path)

        # 每个文件之间添加延迟
        # print(f"等待 {delay_between_files} 秒后发送下一个文件...")
        time.sleep(delay_between_files)





def main():
    process_files_with_delay()


if __name__ == "__main__":
    main()


