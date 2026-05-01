#!/usr/bin/env python3
"""
Robot joint control example (终极序列版：支持安全过渡与多点连续接触)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from aimdk_msgs.msg import JointCommandArray, JointStateArray, JointCommand
from std_msgs.msg import Header
import ruckig
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import threading
import time

subscriber_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

publisher_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

class JointArea(Enum):
    HEAD = 'HEAD'
    ARM = 'ARM'
    WAIST = 'WAIST'
    LEG = 'LEG'

@dataclass
class JointInfo:
    name: str
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float

robot_model: Dict[JointArea, List[JointInfo]] = {
    JointArea.LEG: [
        JointInfo("left_hip_pitch_joint", -2.704, 2.556, 40.0, 4.0),
        JointInfo("left_hip_roll_joint", -0.235, 2.906, 40.0, 4.0),
        JointInfo("left_hip_yaw_joint", -1.684, 3.430, 30.0, 3.0),
        JointInfo("left_knee_joint", 0.0000, 2.4073, 80.0, 8.0),
        JointInfo("left_ankle_pitch_joint", -0.803, 0.453, 40.0, 4.0),
        JointInfo("left_ankle_roll_joint", -0.2625, 0.2625, 20.0, 2.0),
        JointInfo("right_hip_pitch_joint", -2.704, 2.556, 40.0, 4.0),
        JointInfo("right_hip_roll_joint", -2.906, 0.235, 40.0, 4.0),
        JointInfo("right_hip_yaw_joint", -3.430, 1.684, 30.0, 3.0),
        JointInfo("right_knee_joint", 0.0000, 2.4073, 80.0, 8.0),
        JointInfo("right_ankle_pitch_joint", -0.803, 0.453, 40.0, 4.0),
        JointInfo("right_ankle_roll_joint", -0.2625, 0.2625, 20.0, 2.0),
    ],
    JointArea.WAIST: [
        JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
        JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
        JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
    ],
    JointArea.ARM: [
        JointInfo("left_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
        JointInfo("left_shoulder_roll_joint", -0.061, 2.993, 20.0, 2.0),
        JointInfo("left_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("left_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
        JointInfo("left_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("left_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
        JointInfo("left_wrist_roll_joint", -1.571, 0.724, 20.0, 2.0),
        JointInfo("right_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
        JointInfo("right_shoulder_roll_joint", -2.993, 0.061, 20.0, 2.0),
        JointInfo("right_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("right_elbow_joint", -2.3556, 0.0000, 20.0, 2.0),
        JointInfo("right_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("right_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
        JointInfo("right_wrist_roll_joint", -0.724, 1.571, 20.0, 2.0),
    ],
    JointArea.HEAD: [
        JointInfo("head_yaw_joint", -0.366, 0.366, 20.0, 2.0),
        JointInfo("head_pitch_joint", -0.3838, 0.3838, 20.0, 2.0),
    ],
}

class JointControllerNode(Node):
    def __init__(self, node_name: str, sub_topic: str, pub_topic: str, area: JointArea, dofs: int):
        super().__init__(node_name)
        self.joint_info = robot_model[area]
        self.dofs = dofs
        self.ruckig = ruckig.Ruckig(dofs, 0.002) 
        self.input = ruckig.InputParameter(dofs)
        self.output = ruckig.OutputParameter(dofs)
        self.ruckig_initialized = False

        self.input.current_position = [0.0] * dofs
        self.input.current_velocity = [0.0] * dofs
        self.input.current_acceleration = [0.0] * dofs

        # 调小了最大速度和加速度，让动作看起来更优雅、更安全
        self.input.max_velocity = [0.8] * dofs
        self.input.max_acceleration = [0.8] * dofs
        self.input.max_jerk = [15.0] * dofs

        self.sub = self.create_subscription(JointStateArray, sub_topic, self.joint_state_callback, subscriber_qos)
        self.pub = self.create_publisher(JointCommandArray, pub_topic, publisher_qos)

    def joint_state_callback(self, msg: JointStateArray):
        self.ruckig_initialized = True

    def set_target_positions_array(self, target_positions: list):
        if len(target_positions) != self.dofs:
            self.get_logger().error(f"维度错误！需要 {self.dofs} 个角度，但只提供了 {len(target_positions)} 个！")
            return

        self.input.target_position = target_positions
        self.input.target_velocity = [0.0] * self.dofs
        self.input.target_acceleration = [0.0] * self.dofs

        while self.ruckig.update(self.input, self.output) in [ruckig.Result.Working, ruckig.Result.Finished]:
            self.input.current_position = self.output.new_position
            self.input.current_velocity = self.output.new_velocity
            self.input.current_acceleration = self.output.new_acceleration

            tolerance = 1e-4
            reached = all(abs(self.output.new_position[i] - self.input.target_position[i]) < tolerance for i in range(self.dofs))

            cmd = JointCommandArray()
            for i, joint in enumerate(self.joint_info):
                j = JointCommand()
                j.name = joint.name
                j.position = self.output.new_position[i]
                j.velocity = self.output.new_velocity[i]
                j.effort = 0.0
                j.stiffness = joint.kp
                j.damping = joint.kd
                cmd.joints.append(j)

            self.pub.publish(cmd)
            
            if reached:
                break
            
            time.sleep(0.002)


def main(args=None):
    rclpy.init(args=args)

    arm_node = JointControllerNode(
        "arm_node",
        "/aima/hal/joint/arm/state",
        "/aima/hal/joint/arm/command",
        JointArea.ARM,
        14 
    )

    # 1. 定义数据 (基于你的逆解仿真数据)
    # 安全过渡点：大臂微微前抬且外扩，手肘微弯，确保远离胸腔
    SAFE_WAYPOINT = [0.4, 0.3, 0.0, -1.0, 0.0, 0.0, 0.0] 
    
    CONTRA_SHOULDER = [-2.4758, -0.1667, -1.6233, -1.7167, 2.6027, -0.0085, 0.0206]
    MID_CHEST       = [-0.3795, 0.4083, -1.1927, -2.2765, 1.0067, -0.0102, 0.0251]
    LOWER_ABDOMEN   = [-0.1615, 0.1781, -1.4202, -1.3857, 0.9645, -0.0031, 0.0033]

    # 构建带安全过渡的动作队列
    action_sequence = [
        ("【过渡】准备姿态", SAFE_WAYPOINT),
        ("【接触】对侧肩点 (Contra-Shoulder)", CONTRA_SHOULDER),
        ("【过渡】准备姿态", SAFE_WAYPOINT),
        ("【接触】胸前中线 (Mid-Chest)", MID_CHEST),
        ("【过渡】准备姿态", SAFE_WAYPOINT),
        ("【接触】下腹前点 (Lower-Abdomen)", LOWER_ABDOMEN),
        ("【结束】收起手臂", SAFE_WAYPOINT),
    ]

    def sequence_runner():
        """
        使用独立线程运行动作序列，防止阻塞 ROS 2 调度器
        """
        arm_node.get_logger().info("等待 2 秒以确保物理环境稳定...")
        time.sleep(2.0)
        
        arm_node.get_logger().info("🚀 开始执行 [多点防碰撞体表接触] 序列测试...")

        for name, joints in action_sequence:
            arm_node.get_logger().info(f"⏳ 正在前往 -> {name} ...")
            
            # 将 7 自由度补齐为 14 自由度 (右臂保持 0.0)
            target_array = joints + [0.0] * 7
            
            # 阻塞式下发，直到该动作执行完成
            arm_node.set_target_positions_array(target_array)
            
            arm_node.get_logger().info(f"✅ 已到达: {name}")
            
            # 在接触点或过渡点停顿 2.5 秒，方便观察和截图
            time.sleep(2.5) 

        arm_node.get_logger().info("🎉 所有体表接触仿真测试圆满完成！")

    # 2. 启动动作序列线程
    seq_thread = threading.Thread(target=sequence_runner)
    seq_thread.start()

    # 3. 运行 ROS 2 节点
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(arm_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        arm_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
