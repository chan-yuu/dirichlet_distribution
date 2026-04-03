import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D

def plot_dirichlet_with_arrows(alpha, n_samples=200):
    # 1. 生成网格与单纯形坐标
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    X, Y = np.meshgrid(x, y)
    
    mask = X + Y <= 1
    theta1 = X[mask]
    theta2 = Y[mask]
    theta3 = np.clip(1.0 - theta1 - theta2, 0, 1) 
    
    points = np.vstack([theta1, theta2, theta3]).T
    
    # 2. 处理极值与计算 PDF
    eps = 1e-8
    safe_points = np.clip(points, eps, 1 - eps)
    safe_points /= safe_points.sum(axis=1)[:, np.newaxis]
    pdf_values = dirichlet.pdf(safe_points.T, alpha)

    # 3. 绘图准备
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 Dirichlet 分布的散点热力图
    p = ax.scatter(theta1, theta2, theta3, c=pdf_values, cmap='RdYlBu_r', s=10, alpha=0.8)
    
    # 绘制单纯形的几何轮廓（灰色虚线三角形）
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'gray', linestyle='--', lw=1, alpha=0.5)

    # ==== 核心：绘制带有箭头的正交坐标轴 ====
    
    # 设定箭头长度稍微超出单纯形边界 (单纯形边界为 1.0)
    arrow_len = 1.25
    
    # ax.quiver(x, y, z, u, v, w) 表示起点在 (x,y,z)，方向和长度由向量 (u,v,w) 决定
    # arrow_length_ratio 控制箭头头部的大小
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='black', arrow_length_ratio=0.08, lw=1.5) # theta 1 轴
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='black', arrow_length_ratio=0.08, lw=1.5) # theta 2 轴
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='black', arrow_length_ratio=0.08, lw=1.5) # theta 3 轴

    # 将坐标轴标签“钉”在箭头末端稍微靠外的位置
    label_offset = 1.35
    ax.text(label_offset, 0, 0, r'$\theta_1$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    ax.text(0, label_offset, 0, r'$\theta_2$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    ax.text(0, 0, label_offset, r'$\theta_3$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    
    # ==== 视图与排版优化 ====

    # 隐藏系统默认的 3D 边框、刻度和灰色背景板，让自定义的坐标轴更清晰
    ax.set_axis_off() 

    # 留出足够的显示空间，防止箭头和标签被截断
    ax.set_xlim([-0.1, 1.5])
    ax.set_ylim([-0.1, 1.5])
    ax.set_zlim([-0.1, 1.5])
    
    # 强制物理空间比例严格为 1:1:1，保证空间正交性
    ax.set_box_aspect([1, 1, 1]) 

    # 添加标题和颜色条
    ax.set_title(f'Dirichlet Distribution PDF $\\alpha$={alpha}', pad=20, fontsize=15)
    fig.colorbar(p, ax=ax, label='Probability Density', shrink=0.5, pad=0.05)
    
    # 调整到一个能清楚看到三个轴原点的绝佳视角
    ax.view_init(elev=25, azim=35)
    plt.show()

# 运行代码
plot_dirichlet_with_arrows(alpha=[2, 5, 10])