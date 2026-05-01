import mujoco
import mujoco.viewer
import numpy as np

def main():
    # ==========================================
    # 1. 核心表面距离参数 (按需独立调整)
    # ==========================================
    # G1 尺寸比 X2 略小，这里我微调了厚度参数，你可以根据实际包络情况自行修改 (单位: 米)
    D_t_cs = 0.05  # 对侧肩表面厚度 (右肩前凸)
    D_t_mc = 0.08  # 胸前中线表面厚度 (胸甲最凸出)
    D_t_la = 0.07  # 下腹前点表面厚度 (腹部适中)

    # ==========================================
    # 2. 从 XML 加载 G1 模型并提取原生关节坐标
    # ==========================================
    # 🔴 指向 G1 的 XML 文件 (这里使用带有 10kp 或无重力的那个皆可)
    xml_path = "task_g1_23dof.xml"
    print(f"正在加载原生模型: {xml_path} ...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 保持零位姿（默认直立），进行正向解算以刷新空间绝对坐标
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    # 获取躯干 (torso_link) 的全局位姿与旋转矩阵
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_pos = data.xpos[torso_id].copy()
    torso_mat = data.xmat[torso_id].copy().reshape(3, 3)

    # 获取四个基准关节的 ID (用 Roll 关节定义宽度最准确)
    jnt_ls_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
    jnt_rs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")
    jnt_lh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll_joint")
    jnt_rh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll_joint")

    # 获取这四个关节在全局物理世界中的绝对坐标 (Anchor)
    g_pos_ls = data.xanchor[jnt_ls_id].copy()
    g_pos_rs = data.xanchor[jnt_rs_id].copy()
    g_pos_lh = data.xanchor[jnt_lh_id].copy()
    g_pos_rh = data.xanchor[jnt_rh_id].copy()

    # ==========================================
    # 3. 坐标系转换：全局坐标 -> torso_link 局部坐标
    # ==========================================
    def global_to_local(g_pos):
        return torso_mat.T @ (g_pos - torso_pos)

    l_pos_ls = global_to_local(g_pos_ls)
    l_pos_rs = global_to_local(g_pos_rs)
    l_pos_lh = global_to_local(g_pos_lh)
    l_pos_rh = global_to_local(g_pos_rh)

    # ==========================================
    # 4. 计算二维平面的骨架特征量
    # ==========================================
    W_s = abs(l_pos_ls[1] - l_pos_rs[1])       # 肩宽
    P_sm = (l_pos_ls + l_pos_rs) / 2.0         # 肩中点
    P_hm = (l_pos_lh + l_pos_rh) / 2.0         # 胯中点
    H_t = abs(P_sm[2] - P_hm[2])               # 躯干高度

    # ==========================================
    # 5. 基于独立厚度生成三大标志点 (在 torso_link 局部坐标系下)
    # ==========================================
    # 1. 对侧肩点 P_cs: 独立应用 D_t_cs
    P_cs = np.array([P_sm[0] + D_t_cs, P_sm[1] - 0.5 * W_s, P_sm[2]])
    
    # 2. 胸前中线 P_mc: 独立应用 D_t_mc
    P_mc = np.array([P_sm[0] + D_t_mc, P_sm[1], P_sm[2] - 0.25 * H_t])
    
    # 3. 下腹前点 P_la: 独立应用 D_t_la
    P_la = np.array([P_hm[0] + D_t_la, P_hm[1], P_sm[2] - 0.85 * H_t])

    # ==========================================
    # 6. 计算全局坐标 (用于最后的可视化展示)
    # ==========================================
    g_P_cs = torso_pos + torso_mat @ P_cs
    g_P_mc = torso_pos + torso_mat @ P_mc
    g_P_la = torso_pos + torso_mat @ P_la

    # ==========================================
    # 7. 控制台打印：生成你可以直接复制的 XML 格式代码
    # ==========================================
    print("\n" + "="*65)
    print("【第一部分：提取的 G1 机器人原生骨架特征】")
    print(f"  -> 肩宽 W_s = {W_s:.4f} m")
    print(f"  -> 躯干高 H_t = {H_t:.4f} m")
    print(f"  -> [独立参数] 对侧肩表面厚度 D_t_cs = {D_t_cs:.4f} m")
    print(f"  -> [独立参数] 胸前中线表面厚度 D_t_mc = {D_t_mc:.4f} m")
    print(f"  -> [独立参数] 下腹前点表面厚度 D_t_la = {D_t_la:.4f} m")
    print("="*65)
    
    print("\n【第二部分：请将以下代码复制到你的 task_g1_23dof.xml 中】")
    print("🚨 核心注意：为了与 MATLAB 坐标系 0 误差对齐，请务必将其放置在 <body name=\"torso_link\"> 标签内部！\n")
    
    print(f'''      <body name="target_contra_shoulder" pos="{P_cs[0]:.4f} {P_cs[1]:.4f} {P_cs[2]:.4f}">
        <site name="site_contra_shoulder" size="0.01" rgba="0 1 1 0.8" group="1"/>
      </body>
      
      <body name="target_mid_chest" pos="{P_mc[0]:.4f} {P_mc[1]:.4f} {P_mc[2]:.4f}">
        <site name="site_mid_chest" size="0.01" rgba="1 0 1 0.8" group="1"/>
      </body>
      
      <body name="target_lower_abdomen" pos="{P_la[0]:.4f} {P_la[1]:.4f} {P_la[2]:.4f}">
        <site name="site_lower_abdomen" size="0.01" rgba="1 1 0 0.8" group="1"/>
      </body>''')
    print("="*65 + "\n")

    # ==========================================
    # 8. 在 MuJoCo 中动态注入并可视化验证
    # ==========================================
    xml_extension = f"""
    <mujoco>
        <include file="{xml_path}"/>
        <worldbody>
            <site name="vis_cs" pos="{g_P_cs[0]} {g_P_cs[1]} {g_P_cs[2]}" type="sphere" size="0.025" rgba="0 1 1 0.8" />
            <site name="vis_mc" pos="{g_P_mc[0]} {g_P_mc[1]} {g_P_mc[2]}" type="sphere" size="0.025" rgba="1 0 1 0.8" />
            <site name="vis_la" pos="{g_P_la[0]} {g_P_la[1]} {g_P_la[2]}" type="sphere" size="0.025" rgba="1 1 0 0.8" />
        </worldbody>
    </mujoco>
    """

    try:
        vis_model = mujoco.MjModel.from_xml_string(xml_extension)
        vis_data = mujoco.MjData(vis_model)
    except Exception as e:
        print(f"❌ 动态加载可视化模型失败: {e}")
        return

    print("🖥️ 正在启动 MuJoCo 3D 可视化界面供你预览...")
    with mujoco.viewer.launch_passive(vis_model, vis_data) as viewer:
        while viewer.is_running():
            pass

if __name__ == "__main__":
    main()