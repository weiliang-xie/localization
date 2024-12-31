cmake_minimum_required(VERSION 3.0.2)
project(lecd)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wuninitialized")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DPJSRCDIR=\"${PROJECT_SOURCE_DIR}/\")      #定义代码中使用的PJSRCDIR 代表项目的根目录

add_definitions(-DHUMAN_READABLE=0)  # bool: whether to tidy up and print (too detailed) results to screen
add_definitions(-DSAVE_MID_FILE=0)   # bool: whether to save intermediate files (unnecessary to the program)   #这里是将保存中间文件打开
add_definitions(-DPUB_ROS_MSG=0)     # bool: whether to publish message to the program
add_definitions(-DDYNAMIC_THRES=0)   # bool: whether to use dynamic threshold when checking (lift lower bound on the go)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        rosbag
        std_msgs
        tf
        eigen_conversions
        message_generation
        )


find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
#find_package(glog REQUIRED)

find_package(OpenCV 4 REQUIRED)

find_package(Ceres 2 REQUIRED)

# set(CMAKE_PREFIX_PATH "/usr/local")
find_package(PahoMqttCpp REQUIRED)
find_package(nlohmann_json REQUIRED)


include_directories(
        include
        thirdparty
        ${glog_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/matplotlib-cpp
)

catkin_package(
        CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy rosbag std_msgs message_runtime
        DEPENDS EIGEN3 PCL
        INCLUDE_DIRS include
)

add_library(${PROJECT_NAME}lecd
        lecd_match/match/contour.cpp
        lecd_match/match/contour_mng.cpp
        lecd_match/match/contour_db.cpp
        lecd_match/match/correlation.cpp
        )

target_link_libraries(${PROJECT_NAME}lecd
#        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${glog_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        )


add_executable(${PROJECT_NAME}_test
lecd_match/test_localization.cpp
lecd_match/cont2_ros/spinner_ros.cpp
lecd_match/match/evaluator.cpp
lidar_rec/lidar_rec.cpp
)


include_directories(/usr/local/include)

link_directories(/usr/local/lib)

target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}lecd
        ${catkin_LIBRARIES}
        /usr/local/lib/libpaho-mqttpp3.so
        /usr/local/lib/libpaho-mqtt3c.so
        nlohmann_json::nlohmann_json            # 新增：JSON 序列化/反序列化库
        pthread                                 # 新增：线程库（必需）
        )


