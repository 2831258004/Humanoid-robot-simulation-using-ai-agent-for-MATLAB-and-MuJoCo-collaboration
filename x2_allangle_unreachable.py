import mujoco
import numpy as np
import scipy.io
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FormatStrFormatter

def main():
    # ==========================================
    # 1. 批量加载 MATLAB 数据集 (修改为不可达边界数据)
    # ==========================================
    mat_path = 'x2_Unreachable_boundary.mat'
    print(f"正在加载数据集: {mat_path} ...")
    try:
        mat_data = scipy.io.loadmat(mat_path)
        # 【关键修改】：将 valid_pos 替换为 unreachable_pos，valid_angles 替换为 unreachable_angles
        unreachable_pos = mat_data['unreachable_pos']       # 形状: (3, N)
        unreachable_angles = mat_data['unreachable_angles'] # 形状: (5, N)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {mat_path}，请检查路径！")
        return
    except KeyError as e:
         print(f"❌ 数据键值错误，.mat 文件中找不到变量: {e}")
         return

    num_points = unreachable_pos.shape[1]
    print(f"✅ 成功加载，共计 {num_points} 个边界扩展数据点。\n")

    # ==========================================
    # 2. 初始化 MuJoCo 物理引擎 (静默模式)
    # ==========================================
    xml_path = "scene.xml"
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 提前获取各种底层 ID，极大提升循环内执行效率
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_point_out")
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "virtual_tip")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_point_site_out")

    joint_names = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", 
        "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint"
    ]
    
    # 获取关节对应的 qpos 地址索引
    qpos_adrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joint_names]

    # ==========================================
    # 3. 极速验证循环
    # ==========================================
    errors = np.zeros(num_points)
    
    print("🚀 开始极速运动学边界验证推演...")
    start_time = time.time()

    for i in range(num_points):
        # 【关键修改】：提取当前点的目标位置和关节角，使用 unreachable 变量
        target_xyz = unreachable_pos[:, i]
        target_q = unreachable_angles[:, i]

        # A. 放置目标红球
        model.body_pos[target_body_id] = target_xyz

        # B. 强制设定当前关节位置
        for j in range(5):
            data.qpos[qpos_adrs[j]] = target_q[j]

        # C. 瞬间推演正向运动学 (仅更新几何和坐标，不进行动力学计算)
        mujoco.mj_forward(model, data)

        # D. 提取绝对坐标并计算欧氏距离
        pos_tip = data.site_xpos[tip_site_id]
        pos_target = data.site_xpos[target_site_id]
        
        errors[i] = np.linalg.norm(pos_tip - pos_target)

    end_time = time.time()
    print(f"⏱️ 推演完成！耗时: {end_time - start_time:.3f} 秒\n")

    # ==========================================
    # 4. 统计结果分析
    # ==========================================
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # 转换为毫米 (mm) 方便人类阅读
    errors_mm = errors * 1000
    mean_error_mm = mean_error * 1000
    max_error_mm = max_error * 1000

    print("="*45)
    print("【不可达边界测试验证报告】")
    print(f"总测试点数 : {num_points}")
    print(f"最大绝对误差 : {max_error_mm:.6f} 毫米 ({max_error:.8f} 米)")
    print(f"平均绝对误差 : {mean_error_mm:.6f} 毫米 ({mean_error:.8f} 米)")
    print("="*45)

    if max_error_mm < 1.0:
        print("🎉 完美！所有越界点位的数学空间均与底层物理空间严丝合缝对齐！")
    else:
        print("⚠️ 警告：存在大于 1mm 的误差，请结合曲线图定位突变点。")

    # ==========================================
    # 5. 可视化：误差分布直方图与概率密度曲线
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 5.1 绘制直方图 (Density=True 使得面积积分为1，方便叠加概率密度曲线)
    plt.hist(errors_mm, bins=50, density=True, color='#d62728', alpha=0.6, edgecolor='black', linewidth=0.5, label='Error Frequency')
    
    # 5.2 计算并绘制核密度估计曲线 (KDE - Probability Density Function)
    if np.var(errors_mm) > 1e-12:
        kde = gaussian_kde(errors_mm)
        x_vals = np.linspace(min(errors_mm), max(errors_mm), 200)
        plt.plot(x_vals, kde(x_vals), color='darkorange', linewidth=2.5, label='Density Curve (KDE)')
    
    # 5.3 绘制平均误差垂直参考线
    plt.axvline(x=mean_error_mm, color='black', linestyle='--', linewidth=2, 
                label=f'Mean Error: {mean_error_mm:.4f} mm')

    # 5.4 图表装饰与排版
    plt.title('Boundary Point Error Distribution (MATLAB vs MuJoCo)', fontsize=20, fontweight='bold', pad=15)
    plt.xlabel('Kinematic Error Distance (mm)', fontsize=20)
    plt.ylabel('Probability Density', fontsize=20)

    # 更改 X 轴和 Y 轴上【刻度数字】的字体大小 (labelsize)
    plt.tick_params(axis='both', which='major', labelsize=25) 
    
    # 强制设置刻度数字的小数位数
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()