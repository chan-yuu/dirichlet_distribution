import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

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
    arrow_len = 1.25
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='black', arrow_length_ratio=0.08, lw=1.5)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='black', arrow_length_ratio=0.08, lw=1.5)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='black', arrow_length_ratio=0.08, lw=1.5)

    label_offset = 1.35
    ax.text(label_offset, 0, 0, r'$\theta_1$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    ax.text(0, label_offset, 0, r'$\theta_2$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    ax.text(0, 0, label_offset, r'$\theta_3$', fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    
    # ==== 🟢 关键修改：正确启用 3D 网格 ====
    
    # 启用网格并设置样式
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    
    # 设置三个坐标平面的背景为透明
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # 隐藏刻度数字和标签（但保留网格）
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # 移除坐标轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # 留出足够的显示空间
    ax.set_xlim([-0.1, 1.5])
    ax.set_ylim([-0.1, 1.5])
    ax.set_zlim([-0.1, 1.5])
    
    # 强制物理空间比例严格为 1:1:1
    ax.set_box_aspect([1, 1, 1]) 

    ax.set_title(f'Dirichlet Distribution PDF $\\alpha$={alpha}', pad=20, fontsize=15)
    fig.colorbar(p, ax=ax, label='Probability Density', shrink=0.5, pad=0.05)
    
    ax.view_init(elev=25, azim=35)
    # plt.tight_layout()
    plt.show()

# 运行代码
plot_dirichlet_with_arrows(alpha=[2, 5, 10])