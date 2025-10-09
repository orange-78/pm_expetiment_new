import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_handler import DataManager

def plot_grid_graph(lookbacks, steps, metrics,
                    title='Heatmap of MAE by lookback and steps',
                    metric_name='MAE',
                    unit='',
                    scale=1.0,
                    figsize=(6,5),
                    reverse_colorbar_num=False,
                    reverse_colorbar_color=False,
                    cmap='viridis'): 
    
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
    cbar.ax.text(1.5, 1.0, "better", va="center", ha="left", color="black", transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, -0.0, "worse",  va="center", ha="left", color="black", transform=cbar.ax.transAxes)

    ax.set_xlabel("steps")
    ax.set_ylabel("lookback")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

