import folium
import numpy as np
import pandas as pd
#!废弃 不使用

# **文件路径**
csv_filename = "/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_data.csv"

# **读取 CSV 文件**
gps_df = pd.read_csv(csv_filename)

# **直接使用 WGS84 经纬度（不需要 UTM 转换）**
lat_lon_points = list(zip(gps_df["latitude"], gps_df["longitude"]))

# **获取中心点（用于居中地图）**
center_lat, center_lon = gps_df["latitude"].mean(), gps_df["longitude"].mean()

# **创建地图（使用 Google 卫星图）**
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    tiles="Esri.WorldImagery"
)


# **绘制 GPS 轨迹**
folium.PolyLine(lat_lon_points, color="red", weight=2.5, opacity=1).add_to(m)

# **标记起点和终点**
folium.Marker(lat_lon_points[0], popup="起点", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(lat_lon_points[-1], popup="终点", icon=folium.Icon(color="red")).add_to(m)

# **保存地图**
map_filename = "/home/jtcx/remote_control/code/localization/data_pre/gtpose/xuda/gps_trajectory_map.html"
m.save(map_filename)
print(f"✅ 地图已保存为 {map_filename}，可在浏览器中打开！")
