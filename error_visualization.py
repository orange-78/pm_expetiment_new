import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


def calculate_mae_by_step(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    计算每步的累计平均误差
    :param actual: 实际值数组，形状(batchsize, steps, 2)
    :param predicted: 预测值数组，形状(batchsize, steps, 2)
    :return: 每步的累计平均误差数组，形状(batchsize, steps, 2)
    """
    # 计算每个位置的绝对误差
    absolute_errors = np.abs(actual - predicted)
    
    # 对于每一步，计算该步及之前所有步的累计平均误差
    # 使用cumsum计算累计和，然后除以步数得到平均值
    cumsum_errors = np.cumsum(absolute_errors, axis=1)
    
    # 创建步数数组 [1, 2, 3, ..., steps]
    steps = np.arange(1, actual.shape[1] + 1)
    
    # 广播除法：cumsum_errors的形状是(batchsize, steps, 2)
    # steps需要reshape为(1, steps, 1)以便正确广播
    mae_by_step = cumsum_errors / steps.reshape(1, -1, 1)
    
    return mae_by_step


def calculate_mae_of_dataset(mae: np.ndarray) -> np.ndarray:
    """
    计算整个数据集的平均每步MAE
    :param mae: 每步的累计平均误差，形状(batchsize, steps, 2)
    :return: 整个数据集的平均每步MAE，形状(steps, 2)
    """
    return mae.mean(axis=0)


def plot_mae_by_step(
    maes: Dict[str, np.ndarray],
    strlist: Optional[List[str]] = None,
    shape: Optional[Tuple[int, int]] = None,
    orientation: str = 'horizontal',
    scale: float = 1000.0
):
    """
    绘制每步的累计平均误差曲线
    :param maes: 每步的累计平均误差字典,键为模型名称，值为每步的累计误差
    :param strlist: 可选，指定图例显示顺序的模型名称列表
    :param shape: 可选，指定每个子图的大小 (width, height)
    :param orientation: 子图排列方向，'horizontal'(横置)或'vertical'(纵置)
    :param scale: 可选，纵轴数值缩放比例，默认为1.0
    """
    # 确定绘图顺序
    if strlist is not None:
        model_names = [name for name in strlist if name in maes]
    else:
        model_names = list(maes.keys())
    
    # 为每个模型的数据首位添加0，并记录最大步数
    padded_maes = {}
    max_steps = 0
    
    for model_name in model_names:
        mae = maes[model_name]
        # 在首位添加0值
        zero_row = np.zeros((1, mae.shape[1]))
        padded_mae = np.vstack([zero_row, mae])
        padded_maes[model_name] = padded_mae
        max_steps = max(max_steps, padded_mae.shape[0])
    
    # 确定图形大小和布局
    if orientation == 'horizontal':
        nrows, ncols = 1, 2
        if shape is not None:
            figsize = (shape[0] * 2, shape[1])
        else:
            figsize = (14, 5)
        sharex, sharey = False, True
    else:  # vertical
        nrows, ncols = 2, 1
        if shape is not None:
            figsize = (shape[0], shape[1] * 2)
        else:
            figsize = (7, 10)
        sharex, sharey = True, False
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(nrows, ncols, figsize=figsize, 
                                    sharex=sharex, sharey=sharey)
    
    # 计算合适的x轴刻度间隔（从0开始）
    actual_max = max_steps - 1  # 实际最大步数（不含0）
    if actual_max <= 10:
        tick_interval = 1
    elif actual_max <= 20:
        tick_interval = 2
    elif actual_max <= 50:
        tick_interval = 5
    elif actual_max <= 100:
        tick_interval = 10
    elif actual_max <= 200:
        tick_interval = 20
    elif actual_max <= 500:
        tick_interval = 50
    elif actual_max <= 2000:
        tick_interval = 100
    else:
        tick_interval = max(10, actual_max // 10)
    
    # 定义一组区分度高的颜色和线型组合
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 使用tab10色系
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 按指定顺序遍历所有模型
    for idx, model_name in enumerate(model_names):
        mae = padded_maes[model_name]
        steps = np.arange(0, mae.shape[0])  # 从0开始
        
        # 应用缩放比例
        scaled_mae = mae * scale
        
        # 选择颜色、线型和标记
        color = colors[idx % len(colors)]
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        # 绘制第一维度的MAE（仅显示线条，不显示标记点）
        ax1.plot(steps, scaled_mae[:, 0], 
                label=model_name, 
                linewidth=2,
                linestyle=line_style,
                color=color,
                alpha=0.9)
        
        # 绘制第二维度的MAE（仅显示线条，不显示标记点）
        ax2.plot(steps, scaled_mae[:, 1], 
                label=model_name, 
                linewidth=2,
                linestyle=line_style,
                color=color,
                alpha=0.9)
    
    # 设置x轴刻度（从0开始）
    x_ticks = np.arange(0, max_steps, tick_interval)
    if (max_steps - 1) not in x_ticks:
        x_ticks = np.append(x_ticks, max_steps - 1)
    
    # 设置第一个子图
    ax1.set_title('PMX', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (mas)')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(-0.5, max_steps - 0.5)
    
    # 设置第二个子图
    ax2.set_title('PMY', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(-0.5, max_steps - 0.5)
    
    # 根据排列方向设置轴标签和图例
    if orientation == 'horizontal':
        ax1.set_xlabel('Step')
        ax2.set_xlabel('Step')
        # 删除右侧纵轴标示
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelleft=True, labelright=False)
        # 图例放在第二个子图，使用更紧凑的布局
        ax2.legend(loc='best', framealpha=0.9, fontsize=9, 
                  ncol=1 if len(model_names) <= 6 else 2)
        ax1.set_xticks(x_ticks)
        ax2.set_xticks(x_ticks)
    else:  # vertical
        ax2.set_xlabel('Step')
        ax1.set_ylabel('MAE (mas)')
        ax2.set_ylabel('MAE (mas)')
        # 删除上侧横轴标示
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', labelbottom=False)
        # 图例放在第二个子图，使用更紧凑的布局
        ax2.legend(loc='best', framealpha=0.9, fontsize=9,
                  ncol=1 if len(model_names) <= 6 else 2)
        ax2.set_xticks(x_ticks)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    actual = np.random.rand(10, 5, 2)  # 10个样本，5步，2维
    predicted = np.random.rand(10, 5, 2)
    print("actual:\n" + np.array_str(actual))
    print("predicted:\n" + np.array_str(predicted))

    # 计算每步MAE
    mae_by_step = calculate_mae_by_step(actual, predicted)
    print(f"MAE by step shape: {mae_by_step.shape}")  # (10, 5, 2)
    print("MAE by step:\n" + np.array_str(mae_by_step))

    # 计算数据集平均MAE
    dataset_mae = calculate_mae_of_dataset(mae_by_step)
    print(f"Dataset MAE shape: {dataset_mae.shape}")  # (5, 2)
    print("Dataset MAE:\n" + np.array_str(dataset_mae))
    
    # 测试绘图功能
    # 创建多个模型的MAE数据用于演示
    predicted2 = np.random.rand(10, 5, 2)
    mae_by_step2 = calculate_mae_by_step(actual, predicted2)
    dataset_mae2 = calculate_mae_of_dataset(mae_by_step2)
    
    predicted3 = np.random.rand(10, 5, 2)
    mae_by_step3 = calculate_mae_by_step(actual, predicted3)
    dataset_mae3 = calculate_mae_of_dataset(mae_by_step3)
    
    # 绘制多个模型的MAE曲线
    maes_dict = {
        'Model A': dataset_mae,
        'Model B': dataset_mae2,
        'Model C': dataset_mae3
    }
    
    # # 测试1: 默认绘图
    # print("\n测试1: 默认绘图")
    # plot_mae_by_step(maes_dict)
    
    # # 测试2: 指定图例显示顺序
    # print("\n测试2: 指定图例显示顺序 (C, A, B)")
    # plot_mae_by_step(maes_dict, strlist=['Model C', 'Model A', 'Model B'])
    
    # # 测试3: 指定图形大小
    # print("\n测试3: 指定图形大小 (10x6)")
    # plot_mae_by_step(maes_dict, shape=(10, 6))
    
    # 测试4: 同时指定顺序和大小
    print("\n测试4: 同时指定顺序和大小")
    plot_mae_by_step(
        maes_dict, 
        strlist=['Model B', 'Model C', 'Model A'],
        shape=(8, 4)
    )