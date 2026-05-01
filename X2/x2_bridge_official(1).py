import rclpy
from rclpy.node import Node
import mujoco
import numpy as np
import time
from threading import Thread

# 导入官方消息和服务包
from aimdk_msgs.msg import JointCommandArray
from aimdk_msgs.srv import GetMcAction 

class X2OfficialBridge(Node):
    def __init__(self):
        super().__init__('x2_official_bridge')
        
        # 1. 加载 MuJoCo 模型 (指向你的 XML 路径)
        xml_path = "../../pythonsim/models/x2_ultra.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 2. 官方左臂关节名称列表
        self.joint_names = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
            'left_shoulder_yaw_joint', 'left_elbow_joint',
            'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint'
        ]
        
        # 获取 MuJoCo 内部索引映射
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + n) for n in self.joint_names]

        # 3. 初始化控制变量 (防止启动瞬间甩动)
        self.target_q = np.array([0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0]) # 初始准备姿态
        self.kp = np.array([30.0] * 7)
        self.kd = np.array([1.0] * 7)

        # 4. ROS 2 订阅官方控制话题
        self.cmd_sub = self.create_subscription(
            JointCommandArray, 
            '/aima/hal/joint/arm/command', 
            self.cmd_callback, 
            10)
            
        # 5. 提供一个虚假的 MC 状态服务，骗过官方 Demo
        self.get_action_srv = self.create_service(
            GetMcAction, 
            '/aimdk_5Fmsgs/srv/GetMcAction',
            self.get_mc_action_callback
        )
        
        # 启动 1000Hz 物理线程
        self.sim_thread = Thread(target=self.physics_loop)
        self.sim_thread.start()
        self.get_logger().info("✅ X2 官方协议仿真桥接已启动！(缩进修复版)")

    def get_mc_action_callback(self, request, response):
        """ 处理查询模式的请求 """
        # 根据官方代码的要求，把返回值塞进 info 结构体里
        response.info.action_desc = "JOINT_DEFAULT" 
        response.info.status.value = 1  # 随便给个正常状态码
        self.get_logger().info('收到模式查询，已成功“欺骗”官方 Demo！')
        return response

    def cmd_callback(self, msg):
        """ 解析官方指令包 """
        for j_cmd in msg.joints:
            if j_cmd.name in self.joint_names:
                idx = self.joint_names.index(j_cmd.name)
                self.target_q[idx] = j_cmd.position
                # 如果指令里没有给刚度和阻尼，就用默认值
                self.kp[idx] = j_cmd.stiffness if j_cmd.stiffness > 0 else 30.0
                self.kd[idx] = j_cmd.damping if j_cmd.damping > 0 else 1.0

    def physics_loop(self):
        """ 核心 PD 控制循环 """
        while rclpy.ok():
            step_start = time.time()
            
            for i in range(7):
                q_curr = self.data.qpos[self.model.jnt_qposadr[self.joint_ids[i]]]
                v_curr = self.data.qvel[self.model.jnt_dofadr[self.joint_ids[i]]]
                
                torque = self.kp[i] * (self.target_q[i] - q_curr) - self.kd[i] * v_curr
                self.data.ctrl[self.actuator_ids[i]] = torque
            
            mujoco.mj_step(self.model, self.data)
            
            elapsed = time.time() - step_start
            if elapsed < 0.001:
                time.sleep(0.001 - elapsed)

def main():
    rclpy.init()
    node = X2OfficialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
