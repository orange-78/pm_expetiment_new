"""
绘制图像 visualizer.py
"""
import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_handler import DataManager
from tf_singleton import tf
import numpy as np
import matplotlib.pyplot as plt

# 示例调用
# (X_train, y_train), (X_val, y_val), (X_test, y_test), scalers, (train_raw, val_raw, test_raw) = prepare_datasets(...)
# T_test, p_test = get_true_pred_sequences("best_model_weights.h5", X_test, y_test)

def plot_pm(T_test, p_test, i):
    """
    可视化第 i 条数据的 PMX 和 PMY 真值与预测值对比
    
    参数：
        T_test: np.ndarray, shape (n, lookback+steps, 2) 真实序列
        p_test: np.ndarray, shape (n, lookback+steps, 2) 预测序列
        i: int, 要绘制的样本索引
    """
    true_seq = T_test[i]
    pred_seq = p_test[i]
    
    # 时间轴
    timesteps = np.arange(true_seq.shape[0])
    
    # 提取 PMX 和 PMY
    pmx_true, pmy_true = true_seq[:, 0], true_seq[:, 1]
    pmx_pred, pmy_pred = pred_seq[:, 0], pred_seq[:, 1]
    
    # 绘图
    fig, axes= plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # PMX
    axes[0].plot(timesteps, pmx_true, label="True PMX", color="blue")
    axes[0].plot(timesteps, pmx_pred, label="Pred PMX", color="red", linestyle="--")
    axes[0].set_ylabel("PMX")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # PMY
    axes[1].plot(timesteps, pmy_true, label="True PMY", color="blue")
    axes[1].plot(timesteps, pmy_pred, label="Pred PMY", color="red", linestyle="--")
    axes[1].set_ylabel("PMY")
    axes[1].set_xlabel("Time step")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_grid_graph(lookbacks, steps, metrics,
                    title='Heatmap of MAE by lookback and steps',
                    metric_name='MAE',
                    unit='',
                    scale=1.0,
                    figsize=(6,5),
                    reverse_colorbar_num=False,
                    reverse_colorbar_color=False,
                    cmap='viridis',
                    font_size=28): 
    
    # 调整字号
    plt.rcParams.update({'font.size': font_size})

    # 缩放数据
    metrics = [m * scale for m in metrics]

    # 获取唯一的标签
    unique_lookbacks = sorted(set(lookbacks))
    unique_steps = sorted(set(steps))

    # 创建 lookback->行索引、steps->列索引的映射
    lookback_idx = {lb: i for i, lb in enumerate(unique_lookbacks)}
    steps_idx = {st: j for j, st in enumerate(unique_steps)}

    # 初始化矩阵，用 nan 表示缺失
    heatmap = np.full((len(unique_lookbacks), len(unique_steps)), np.nan)

    # 填充数据
    for lb, st, val in zip(lookbacks, steps, metrics):
        i, j = lookback_idx[lb], steps_idx[st]
        heatmap[i, j] = val

    # 绘制图像
    fig, ax = plt.subplots(figsize=figsize)

    # 如果传入的是字符串，转为 colormap 对象
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='white')  # nan 填充为白色

    # 逆转colorbar颜色，使大数对应暗色，小数对应亮色
    if reverse_colorbar_color:
        cmap = cmap.reversed()   # 反向颜色映射

    im = ax.imshow(heatmap, cmap=cmap, origin='lower')

    # 设置坐标轴刻度为对应标签
    ax.set_xticks(range(len(unique_steps)))
    ax.set_xticklabels(unique_steps)
    ax.set_yticks(range(len(unique_lookbacks)))
    ax.set_yticklabels(unique_lookbacks)

    # 添加 colorbar，并让它和图像高度匹配
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(metric_name)

    # 设置带单位的标签
    label = metric_name if not unit else f"{metric_name} ({unit})"
    cbar.set_label(label)

    # 倒置 colorbar，使顶部为小值（better），底部为大值（worse）
    if reverse_colorbar_num:
        cbar.ax.invert_yaxis()

    # 在 colorbar 上添加 better / worse
    cbar.ax.text(1.5, 1.05, "better", va="center", ha="left", color="black", transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, -0.05, "worse",  va="center", ha="left", color="black", transform=cbar.ax.transAxes)

    ax.set_xlabel("steps")
    ax.set_ylabel("lookback")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

