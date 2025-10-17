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
from datetime import datetime, timedelta

# 示例调用
# (X_train, y_train), (X_val, y_val), (X_test, y_test), scalers, (train_raw, val_raw, test_raw) = prepare_datasets(...)
# T_test, p_test = get_true_pred_sequences("best_model_weights.h5", X_test, y_test)

def plot_pm(
    T_test,
    p_test,
    i=0,
    start_date=None,
    date_format="%Y-%m-%d",
    labels=("True PM", "Pred PM"),
):
    """
    可视化第 i 条数据的 PMX 和 PMY 真值与预测值对比（支持左对齐与自定义图例）

    参数：
        T_test: np.ndarray, shape (n, L1, 2) 或 (L1, 2)，真实序列
        p_test: np.ndarray, shape (n, L2, 2) 或 (L2, 2)，预测序列（可不等长）
        i: int, 样本索引，仅当输入为 3D 时有效
        start_date: 可选 datetime 或 str, 若提供则横坐标按日期计算
        date_format: 日期格式（仅当 start_date 为 str 时使用）
        labels: tuple(str, str)，图例前缀，例如 ('Observed', 'Forecasted')
    """

    # ==== 支持 2D 或 3D 输入 ====
    if T_test.ndim == 3:
        true_seq = T_test[i]
    elif T_test.ndim == 2:
        true_seq = T_test
    else:
        raise ValueError("T_test 必须是 shape (n, L, 2) 或 (L, 2)")

    if p_test.ndim == 3:
        pred_seq = p_test[i]
    elif p_test.ndim == 2:
        pred_seq = p_test
    else:
        raise ValueError("p_test 必须是 shape (n, L, 2) 或 (L, 2)")

    len_true = true_seq.shape[0]
    len_pred = pred_seq.shape[0]
    max_len = max(len_true, len_pred)

    # ==== 左对齐 ====
    def left_align(arr, target_len):
        """左对齐序列，用 NaN 填充右侧"""
        pad_total = target_len - len(arr)
        if pad_total <= 0:
            return arr[:target_len]
        return np.pad(arr, ((0, pad_total), (0, 0)), mode="constant", constant_values=np.nan)

    true_seq_aligned = left_align(true_seq, max_len)
    pred_seq_aligned = left_align(pred_seq, max_len)

    pmx_true, pmy_true = true_seq_aligned[:, 0], true_seq_aligned[:, 1]
    pmx_pred, pmy_pred = pred_seq_aligned[:, 0], pred_seq_aligned[:, 1]

    # ==== 横坐标 ====
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, date_format)
        time_axis = [start_date + timedelta(days=i) for i in range(max_len)]
        x_label = "Date"
    else:
        time_axis = np.arange(max_len)
        x_label = "Time step"

    # ==== 绘图 ====
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # PMX
    axes[0].plot(time_axis, pmx_true, label=f"{labels[0]} PMX", color="blue")
    axes[0].plot(time_axis, pmx_pred, label=f"{labels[1]} PMX", color="red", linestyle="--")
    axes[0].set_ylabel("PMX")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # PMY
    axes[1].plot(time_axis, pmy_true, label=f"{labels[0]} PMY", color="blue")
    axes[1].plot(time_axis, pmy_pred, label=f"{labels[1]} PMY", color="red", linestyle="--")
    axes[1].set_ylabel("PMY")
    axes[1].set_xlabel(x_label)
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
                    font_size=28,
                    vrange=None): 
    """
    参数说明:
    vrange: tuple or None
        - None: 自动使用数据的最小值和最大值
        - (vmin, vmax): 指定colorbar的范围
          如果数据超出此范围,则自动扩展到数据实际范围
    """
    
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

    # 初始化矩阵,用 nan 表示缺失
    heatmap = np.full((len(unique_lookbacks), len(unique_steps)), np.nan)

    # 填充数据
    for lb, st, val in zip(lookbacks, steps, metrics):
        i, j = lookback_idx[lb], steps_idx[st]
        heatmap[i, j] = val

    # 确定 colorbar 的范围
    data_min = np.nanmin(heatmap)
    data_max = np.nanmax(heatmap)
    
    if vrange is None:
        # 不指定范围,使用数据范围
        vmin, vmax = data_min, data_max
    else:
        # 指定了范围
        vmin_param, vmax_param = vrange
        # 如果数据超出参数范围,则以数据为准
        vmin = min(vmin_param, data_min)
        vmax = max(vmax_param, data_max)

    # 绘制图像
    fig, ax = plt.subplots(figsize=figsize)

    # 如果传入的是字符串,转为 colormap 对象
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='white')  # nan 填充为白色

    # 逆转colorbar颜色,使大数对应暗色,小数对应亮色
    if reverse_colorbar_color:
        cmap = cmap.reversed()   # 反向颜色映射

    im = ax.imshow(heatmap, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    # 设置坐标轴刻度为对应标签
    ax.set_xticks(range(len(unique_steps)))
    ax.set_xticklabels(unique_steps)
    ax.set_yticks(range(len(unique_lookbacks)))
    ax.set_yticklabels(unique_lookbacks)

    # 添加 colorbar,并让它和图像高度匹配
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(metric_name)

    # 设置带单位的标签
    label = metric_name if not unit else f"{metric_name} ({unit})"
    cbar.set_label(label)

    # 倒置 colorbar,使顶部为小值(better),底部为大值(worse)
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
