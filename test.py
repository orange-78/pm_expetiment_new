# import sys

# # 检查 TensorFlow 版本
# try:
#     import tensorflow as tf
#     print("TensorFlow version:", tf.__version__)
# except ImportError:
#     print("TensorFlow not installed")

# # 检查 keras 独立包版本
# try:
#     import keras
#     print("Standalone keras version:", keras.__version__)
# except ImportError:
#     print("Standalone keras not installed")

# # 检查 tf.keras 版本（它是 TensorFlow 自带的 keras）
# try:
#     print("tf.keras version:", tf.keras.__version__)
# except Exception as e:
#     print("tf.keras not available:", e)

# # 输出 Python 版本
# print("Python version:", sys.version)

# import keras
# import sys
# import os

# print("==== Keras 环境检测 ====")

# # keras 包版本
# print("keras.__version__:", getattr(keras, "__version__", "N/A"))

# # keras 模块文件路径
# print("keras.__file__:", keras.__file__)

# # keras 模块所在的顶层包
# print("keras package path:", os.path.dirname(keras.__file__))

# # Python 环境信息
# print("Python executable:", sys.executable)
# print("Python version:", sys.version)

# # 尝试打印 Functional 类的路径
# try:
#     from keras.engine.functional import Functional
#     print("Functional class (old API) loaded from:", Functional)
# except Exception as e1:
#     print("旧API Functional 不存在:", e1)

# try:
#     from keras.src.engine.functional import Functional
#     print("Functional class (new API) loaded from:", Functional)
# except Exception as e2:
#     print("新API Functional 不存在:", e2)

import json
import os
from typing import List, Optional
import numpy as np
from tabulate import tabulate


def extract_and_print_mae_table(
    data_path: str,
    indices: List[int],
    table_format: str = 'grid',
    show_stats: bool = True,
    show_rms: bool = False
):
    """
    从JSON文件中提取指定索引的MAE数据并打印为表格
    
    :param data_path: JSON数据文件路径
    :param indices: 要提取的索引列表 [a1, a2, ..., ak]
    :param table_format: 表格格式，可选: 'grid', 'simple', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex'
    :param show_stats: 是否显示统计信息（平均值、最大值、最小值）
    :param show_rms: 是否显示RMS（方均根）表格，即sqrt(PMX^2 + PMY^2)
    """
    
    print("=" * 80)
    print("📊 MAE数据提取与表格展示")
    print("=" * 80)
    
    # === 1️⃣ 读取JSON文件 ===
    print(f"\n📂 读取数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在: {data_path}")
        return
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # 转换为numpy数组
        maes_dict = {
            label: np.array(values) 
            for label, values in saved_data['maes_dict'].items()
        }
        labels = saved_data['labels']
        
        print(f"✅ 成功加载数据")
        print(f"   模型数量: {len(maes_dict)}")
        print(f"   标签: {labels}")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # === 2️⃣ 提取数据 ===
    print(f"\n🔍 提取索引: {indices}")
    
    # 准备表格数据
    table_data = []
    headers = ['Index', 'Model']
    
    # 为每个索引添加PMX和PMY列
    for idx in indices:
        headers.extend([f'Step {idx} PMX', f'Step {idx} PMY'])
    
    # 遍历每个模型
    for label in labels:
        mae_array = maes_dict[label]  # 形状: (m, 2)
        m = mae_array.shape[0]
        
        row = ['-', label]  # 第一列为索引占位，第二列为模型名称
        
        # 提取每个指定索引的值
        for idx in indices:
            if idx - 1 < m:
                # 索引有效，提取PMX和PMY
                pmx = mae_array[idx - 1, 0]
                pmy = mae_array[idx - 1, 1]
                row.extend([f'{pmx:.4f}', f'{pmy:.4f}'])
            else:
                # 索引超出范围，填充空值
                row.extend(['-', '-'])
        
        table_data.append(row)
    
    # === 3️⃣ 打印主表格 ===
    print(f"\n{'='*80}")
    print("📋 MAE数据表格 (PMX和PMY分量)")
    print(f"{'='*80}\n")
    
    print(tabulate(table_data, headers=headers, tablefmt=table_format))
    
    # === 3.5️⃣ 打印RMS表格（可选）===
    if show_rms:
        print(f"\n{'='*80}")
        print("📊 RMS表格 (方均根: √(PMX² + PMY²))")
        print(f"{'='*80}\n")
        
        rms_table_data = []
        rms_headers = ['Index', 'Model']
        
        # 为每个索引添加RMS列
        for idx in indices:
            rms_headers.append(f'Step {idx} RMS')
        
        # 遍历每个模型计算RMS
        for label in labels:
            mae_array = maes_dict[label]  # 形状: (m, 2)
            m = mae_array.shape[0]
            
            row = ['-', label]
            
            # 计算每个指定索引的RMS值
            for idx in indices:
                if idx - 1 < m:
                    pmx = mae_array[idx - 1, 0]
                    pmy = mae_array[idx - 1, 1]
                    rms = np.sqrt(pmx**2 + pmy**2)
                    row.append(f'{rms:.4f}')
                else:
                    row.append('-')
            
            rms_table_data.append(row)
        
        print(tabulate(rms_table_data, headers=rms_headers, tablefmt=table_format))
        
        # RMS统计信息
        if show_stats:
            print(f"\n{'='*80}")
            print("📈 RMS统计信息")
            print(f"{'='*80}\n")
            
            for idx in indices:
                print(f"--- Step {idx} ---")
                
                # 收集该索引的所有RMS值
                rms_values = []
                
                for label in labels:
                    mae_array = maes_dict[label]
                    m = mae_array.shape[0]
                    
                    if idx - 1 < m:
                        pmx = mae_array[idx - 1, 0]
                        pmy = mae_array[idx - 1, 1]
                        rms = np.sqrt(pmx**2 + pmy**2)
                        rms_values.append(rms)
                
                if rms_values:
                    rms_arr = np.array(rms_values)
                    
                    stats_data = [
                        ['RMS', f'{rms_arr.mean():.4f}', f'{rms_arr.min():.4f}', 
                         f'{rms_arr.max():.4f}', f'{rms_arr.std():.4f}']
                    ]
                    
                    stats_headers = ['Metric', 'Mean', 'Min', 'Max', 'Std']
                    print(tabulate(stats_data, headers=stats_headers, tablefmt=table_format))
                else:
                    print("  (所有模型在此索引处均无数据)")
                
                print()
    
    # === 4️⃣ 打印统计信息（可选）===
    if show_stats and not show_rms:
        print(f"\n{'='*80}")
        print("📈 统计信息")
        print(f"{'='*80}\n")
        
        for idx in indices:
            print(f"--- Step {idx} ---")
            
            # 收集该索引的所有有效值
            pmx_values = []
            pmy_values = []
            
            for label in labels:
                mae_array = maes_dict[label]
                m = mae_array.shape[0]
                
                if idx - 1 < m:
                    pmx_values.append(mae_array[idx - 1, 0])
                    pmy_values.append(mae_array[idx - 1, 1])
            
            if pmx_values:
                pmx_arr = np.array(pmx_values)
                pmy_arr = np.array(pmy_values)
                
                stats_data = [
                    ['PMX', f'{pmx_arr.mean():.4f}', f'{pmx_arr.min():.4f}', 
                     f'{pmx_arr.max():.4f}', f'{pmx_arr.std():.4f}'],
                    ['PMY', f'{pmy_arr.mean():.4f}', f'{pmy_arr.min():.4f}', 
                     f'{pmy_arr.max():.4f}', f'{pmy_arr.std():.4f}']
                ]
                
                stats_headers = ['Metric', 'Mean', 'Min', 'Max', 'Std']
                print(tabulate(stats_data, headers=stats_headers, tablefmt=table_format))
            else:
                print("  (所有模型在此索引处均无数据)")
            
            print()
    
    # === 5️⃣ 打印摘要信息 ===
    print(f"{'='*80}")
    print("📊 提取摘要")
    print(f"{'='*80}")
    print(f"✓ 请求提取索引: {indices}")
    print(f"✓ 模型数量: {len(labels)}")
    
    # 统计每个索引的有效数据数量
    for idx in indices:
        valid_count = sum(1 for label in labels if idx - 1 < maes_dict[label].shape[0])
        print(f"✓ Step {idx}: {valid_count}/{len(labels)} 个模型有数据")
    
    print(f"{'='*80}\n")
    
    return table_data, headers


# === 使用示例 ===
if __name__ == "__main__":
    # # 示例1: 显示PMX和PMY分量表格
    # print("\n" + "="*80)
    # print("示例1: 显示PMX和PMY分量")
    # print("="*80 + "\n")
    # extract_and_print_mae_table(
    #     data_path='data/predicts/mae_figure_data_100d copy.json',
    #     indices=[100, 365, 600, 800, 1100],
    #     table_format='fancy_grid',
    #     show_stats=True,
    #     show_rms=False
    # )
    
    # 示例2: 显示RMS表格
    print("\n" + "="*80)
    print("示例2: 显示RMS（方均根）")
    print("="*80 + "\n")
    extract_and_print_mae_table(
        data_path='data/predicts/mae_figure_data_100d.json',
        indices=[100, 365, 600, 800, 1100],
        table_format='fancy_grid',
        show_stats=True,
        show_rms=True
    )