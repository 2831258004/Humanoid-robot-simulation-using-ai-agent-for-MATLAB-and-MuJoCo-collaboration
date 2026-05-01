#!/bin/bash
# 1. 加载 ROS 2 Humble 基础环境
source /opt/ros/humble/setup.bash
source install/local_setup.bash
# 2. 加载智元 SDK 官方协议
# 自动寻找你编译好的 install 路径
SCRIPT_DIR="/opt/Data/WangChenwei/x2_sim_backend/ros2/aimdk"
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
    echo "智元 SDK 协议已加载！"
else
    echo "警告：未找到 SDK install 文件夹，请确认 colcon build 是否成功。"
fi

# 3. 设置 PYTHONPATH 确保 Conda 3.10 能找到 ROS 2 库
export PYTHONPATH=$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages
