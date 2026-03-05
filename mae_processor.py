import json
import pickle
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import os

from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate

from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, load_config
from error_visualization import calculate_mae_by_step, calculate_mae_of_dataset, plot_mae_by_step

from main import ExperimentRunner, select_model_file


def cal_draw_mae(model_paths: str = None, 
                 labels: List[str] = None,
                 config_path: str = None,
                 save_fig: bool = True,
                 mode: str = 'run',
                 data_path: str = None,
                 data_slice: int = None):
    """
    计算并绘制 MAE(平均绝对误差)
    
    :param model_paths: 模型路径列表,默认为 None(将弹出选择界面)
    :param labels: 标签列表
    :param config_path: 配置文件路径,默认为None
    :param save_fig: 是否保存图像,默认为 True
    :param mode: 运行模式, 'run' = 运行模型并保存数据, 'load' = 读取已有数据
    :param data_path: 数据文件路径,默认为 'data/mae_results.json' (支持 .json 或 .pkl)
    """
    
    print("=" * 60)
    print("🚀 开始执行 cal_draw_mae()")
    print("=" * 60)
    
    # 设置默认数据路径
    if data_path is None:
        data_path = 'data/mae_results.json'
    
    # 判断文件格式
    is_json = data_path.endswith('.json')
    is_pickle = data_path.endswith('.pkl') or data_path.endswith('.pickle')
    
    # === 模式选择 ===
    if mode == 'load':
        # 读取模式
        print(f"📂 读取模式: 从 {data_path} 加载数据...")
        
        if not os.path.exists(data_path):
            print(f"❌ 错误: 数据文件不存在: {data_path}")
            return
        
        try:
            if is_json:
                # JSON 格式
                with open(data_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                # 将列表转换回 numpy 数组
                maes_dict = {
                    label: np.array(values) 
                    for label, values in saved_data['maes_dict'].items()
                }
                labels = saved_data['labels']
                
            elif is_pickle:
                # Pickle 格式
                with open(data_path, 'rb') as f:
                    saved_data = pickle.load(f)
                
                maes_dict = saved_data['maes_dict']
                labels = saved_data['labels']
            
            else:
                print(f"❌ 错误: 不支持的文件格式,请使用 .json 或 .pkl")
                return
            
            print(f"✅ 成功加载数据")
            print(f"   模型数量: {len(maes_dict)}")
            print(f"   标签: {labels}")
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return
    
    elif mode == 'run':
        # 运行模式
        print("🏃 运行模式: 执行模型推理...")
        
        # === 1️⃣ 选择或指定模型路径 ===
        if config_path:
            data_cfg, model_cfg, training_cfg = load_config(config_path)
            runner_default = ExperimentRunner(data_config=data_cfg, model_config=model_cfg, training_config=training_cfg)
        else:
            runner_default = ExperimentRunner()
        
        if not model_paths:
            # 交互式选择单个模型
            model_path = select_model_file(
                runner_default.data_config.model_target_dir, 
                max_depth=7
            )
            if not model_path:
                print("❌ 未选择模型")
                return
            model_paths = [model_path]
        
        # 验证所有模型路径
        valid_model_paths = []
        for path in model_paths:
            if os.path.exists(path):
                valid_model_paths.append(path)
            else:
                print(f"⚠️  警告: 模型路径不存在,已跳过: {path}")
        
        if not valid_model_paths:
            print("❌ 没有有效的模型路径")
            return
        
        print(f"📁 将处理 {len(valid_model_paths)} 个模型")
        
        # === 2️⃣ 处理标签列表 ===
        if labels is None:
            # 使用模型文件名作为标签
            labels = [os.path.basename(path).replace('.keras', '').replace('.h5', '') 
                      for path in valid_model_paths]
        elif len(labels) != len(valid_model_paths):
            print(f"⚠️  警告: 标签数量({len(labels)})与模型数量({len(valid_model_paths)})不匹配,使用默认标签")
            labels = [os.path.basename(path).replace('.keras', '').replace('.h5', '') 
                      for path in valid_model_paths]
        
        # === 3️⃣ 对每个模型进行推理和MAE计算 ===
        maes_dict = {}
        
        for idx, (model_path, label) in enumerate(zip(valid_model_paths, labels)):
            print(f"\n{'='*60}")
            print(f"📊 处理模型 [{idx+1}/{len(valid_model_paths)}]: {label}")
            print(f"   路径: {model_path}")
            print(f"{'='*60}")
            
            # 测试模型并获取预测结果
            print("🔄 正在进行模型推理...")
            test_results = runner_default.test_model(
                model_path=model_path,
                do_predict=[0, 0, 1],  # 仅预测测试集
                print_summary=True
            )
            
            # 提取实际值和预测值
            actual = test_results['ground_truth']['test']  # 形状: (batchsize, steps, 2)
            predicted = test_results['predictions']['test']  # 形状: (batchsize, steps, 2)
            
            print(f"✅ 推理完成")
            print(f"   实际值形状: {actual.shape}")
            print(f"   预测值形状: {predicted.shape}")
            
            # 计算MAE
            print("📊 计算每步MAE...")
            mae_by_step = calculate_mae_by_step(actual, predicted)
            if data_slice:
                dataset_mae = calculate_mae_of_dataset(mae_by_step[:data_slice])
            else:
                dataset_mae = calculate_mae_of_dataset(mae_by_step)
            
            # 保存到字典
            maes_dict[label] = dataset_mae
            
            # 打印详细的MAE统计信息
            print(f"\n📈 MAE统计信息 ({label}):")
            print(f"   PMX - 最小MAE: {dataset_mae[:, 0].min():.4f} mas")
            print(f"   PMX - 最大MAE: {dataset_mae[:, 0].max():.4f} mas")
            print(f"   PMX - 平均MAE: {dataset_mae[:, 0].mean():.4f} mas")
            print(f"   PMY - 最小MAE: {dataset_mae[:, 1].min():.4f} mas")
            print(f"   PMY - 最大MAE: {dataset_mae[:, 1].max():.4f} mas")
            print(f"   PMY - 平均MAE: {dataset_mae[:, 1].mean():.4f} mas")
        
        # === 保存计算结果 ===
        print("\n" + "=" * 60)
        print(f"💾 保存MAE数据到: {data_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        try:
            if is_json:
                # JSON 格式 - 需要将 numpy 数组转为列表
                saved_data = {
                    'maes_dict': {
                        label: mae_values.tolist() 
                        for label, mae_values in maes_dict.items()
                    },
                    'labels': labels,
                    'model_paths': valid_model_paths
                }
                
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(saved_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 数据已保存为 JSON 格式")
                
            elif is_pickle:
                # Pickle 格式 - 可以直接保存 numpy 数组
                saved_data = {
                    'maes_dict': maes_dict,
                    'labels': labels,
                    'model_paths': valid_model_paths
                }
                
                with open(data_path, 'wb') as f:
                    pickle.dump(saved_data, f)
                
                print(f"✅ 数据已保存为 Pickle 格式")
            
            else:
                print(f"⚠️  警告: 不支持的文件格式,使用默认 JSON 格式保存")
                data_path = data_path.rsplit('.', 1)[0] + '.json'
                
                saved_data = {
                    'maes_dict': {
                        label: mae_values.tolist() 
                        for label, mae_values in maes_dict.items()
                    },
                    'labels': labels,
                    'model_paths': valid_model_paths
                }
                
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(saved_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 数据已保存为 JSON 格式至: {data_path}")
            
        except Exception as e:
            print(f"⚠️  警告: 保存数据失败: {e}")
    
    else:
        print(f"❌ 错误: 无效的模式 '{mode}', 请使用 'run' 或 'load' 或 'grid'")
        return
    
    # === 4️⃣ 统一绘制所有模型的MAE曲线 ===
    print("\n" + "=" * 60)
    print("🎨 绘制所有模型的MAE对比曲线...")
    print("=" * 60)
    
    plot_mae_by_step(
        maes_dict,
        strlist=labels,  # 按照输入顺序显示图例
        shape=(7, 5)
    )
    
    # === 5️⃣ 保存图像(可选)===
    if save_fig:
        # 生成包含所有模型名称的文件名
        if len(labels) == 1:
            save_name = f"mae_{labels[0]}"
        else:
            save_name = f"mae_comparison_{len(labels)}models"
        
        save_path = f"figures/{save_name}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图像已保存至: {save_path}")
    
    print("\n" + "=" * 60)
    print("🎯 MAE计算和绘制流程已完成")
    print(f"   共处理 {len(maes_dict)} 个模型")
    print("=" * 60)
    
    return maes_dict  # 返回结果供进一步使用

def extract_and_print_mae_table(
    data_path: str,
    indices: List[int],
    table_format: str = 'grid',
    show_stats: bool = True,
    show_rms: bool = True,
    save_path: str = None
):
    """
    从JSON文件中提取指定索引的MAE数据并打印为表格
    
    :param data_path: JSON数据文件路径
    :param indices: 要提取的索引列表 [a1, a2, ..., ak]
    :param table_format: 表格格式，可选: 'grid', 'simple', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex'
    :param show_stats: 是否显示统计信息（平均值、最大值、最小值）
    :param show_rms: 是否显示RMS（方均根）表格，即sqrt(PMX^2 + PMY^2)
    :param save_path: 保存打印内容的txt文件路径（可选）
    """
    
    # === 创建双输出打印函数 ===
    file_handle = None
    if save_path:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            file_handle = open(save_path, 'w', encoding='utf-8')
        except Exception as e:
            print(f"⚠️ 警告: 无法创建保存文件 {save_path}: {e}")
            file_handle = None
    
    def dual_print(text=""):
        """同时打印到控制台和文件"""
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    try:
        dual_print("=" * 80)
        dual_print("📊 MAE数据提取与表格展示")
        dual_print("=" * 80)
        
        # === 1️⃣ 读取JSON文件 ===
        dual_print(f"\n📂 读取数据文件: {data_path}")
        
        if not os.path.exists(data_path):
            dual_print(f"❌ 错误: 文件不存在: {data_path}")
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
            
            dual_print(f"✅ 成功加载数据")
            dual_print(f"   模型数量: {len(maes_dict)}")
            dual_print(f"   标签: {labels}")
            
        except Exception as e:
            dual_print(f"❌ 读取文件失败: {e}")
            return
        
        # === 2️⃣ 提取数据 ===
        dual_print(f"\n🔍 提取索引: {indices}")
        
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
        dual_print(f"\n{'='*80}")
        dual_print("📋 MAE数据表格 (PMX和PMY分量)")
        dual_print(f"{'='*80}\n")
        
        table_str = tabulate(table_data, headers=headers, tablefmt=table_format)
        dual_print(table_str)
        
        # === 3.5️⃣ 打印RMS表格（可选）===
        if show_rms:
            dual_print(f"\n{'='*80}")
            dual_print("📊 RMS表格 (方均根: √(PMX² + PMY²))")
            dual_print(f"{'='*80}\n")
            
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
            
            rms_table_str = tabulate(rms_table_data, headers=rms_headers, tablefmt=table_format)
            dual_print(rms_table_str)
            
            # RMS统计信息
            if show_stats:
                dual_print(f"\n{'='*80}")
                dual_print("📈 RMS统计信息")
                dual_print(f"{'='*80}\n")
                
                for idx in indices:
                    dual_print(f"--- Step {idx} ---")
                    
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
                        stats_str = tabulate(stats_data, headers=stats_headers, tablefmt=table_format)
                        dual_print(stats_str)
                    else:
                        dual_print("  (所有模型在此索引处均无数据)")
                    
                    dual_print()
        
        # === 4️⃣ 打印统计信息（可选）===
        if show_stats and not show_rms:
            dual_print(f"\n{'='*80}")
            dual_print("📈 统计信息")
            dual_print(f"{'='*80}\n")
            
            for idx in indices:
                dual_print(f"--- Step {idx} ---")
                
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
                    stats_str = tabulate(stats_data, headers=stats_headers, tablefmt=table_format)
                    dual_print(stats_str)
                else:
                    dual_print("  (所有模型在此索引处均无数据)")
                
                dual_print()
        
        # === 5️⃣ 打印摘要信息 ===
        dual_print(f"{'='*80}")
        dual_print("📊 提取摘要")
        dual_print(f"{'='*80}")
        dual_print(f"✓ 请求提取索引: {indices}")
        dual_print(f"✓ 模型数量: {len(labels)}")
        
        # 统计每个索引的有效数据数量
        for idx in indices:
            valid_count = sum(1 for label in labels if idx - 1 < maes_dict[label].shape[0])
            dual_print(f"✓ Step {idx}: {valid_count}/{len(labels)} 个模型有数据")
        
        dual_print(f"{'='*80}\n")
        
        if save_path and file_handle:
            dual_print(f"💾 输出已保存到: {save_path}")
        
        return table_data, headers
    
    finally:
        # 确保文件被正确关闭
        if file_handle:
            file_handle.close()


def extend_data(src_path: str, dst_path: str, label: str):
    # 读取源 list
    with open(src_path, "r", encoding="utf-8") as f:
        src_list = json.load(f)

    if not isinstance(src_list, list):
        raise ValueError("a/b.json 必须是 list")

    # 读取目标 dict
    with open(dst_path, "r", encoding="utf-8") as f:
        dst_data = json.load(f)

    if not isinstance(dst_data.get("maes_dict"), dict):
        raise ValueError('"maes_dict" 不存在或不是 dict')

    # 创建 ls+ar 并赋值
    dst_data["maes_dict"][label] = src_list
    dst_data["labels"].append(label)
    dst_data["model_paths"].append(src_path)

    # 写回
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(dst_data, f, ensure_ascii=False, indent=4)

    print("写入完成")


if __name__ == "__main__":

    main_data_file = "data/predicts/new/mae_data.json"

    # 运行MAE计算和绘制程序
    mode = 'run'  # 'run' or 'load'
    models = [
        "data/models_reproduce/mse-int4/1400_100/model-bestmetric.keras",
        "data/models_reproduce/mse-int4/800_400/model-bestmetric.keras",
        "data/models_reproduce/mse-int4/800_600/model-bestmetric.keras",
        "data/models_reproduce/mse-int4/1200_900/model-bestmetric.keras",
        "data/models_reproduce/mse-int4/1200_1100/model-bestmetric.keras"
    ]
    labels = [
        "100",
        "400",
        "600",
        "900",
        "1100"
    ]
    cal_draw_mae(model_paths=models,
                    labels=labels,
                    config_path='config.json',
                    save_fig=False,
                    mode='run',
                    data_path=main_data_file,
                    data_slice=100)
    
    
    # 合并不同来源 MAE by days 数据
    lsar_label = 'LS+AR'
    lsar_model = 'data/models_baseline/ls_ar/newsliding_window_test/fit_periods/mae_result1100_100d.json'
    extend_data(
        src_path=lsar_model,
        dst_path=main_data_file,
        label=lsar_label
    )
    labels.append(lsar_label)
    ba_label = 'BulletinA'
    ba_model = 'data/models_baseline/IERS/BulletinA16-25/mae_result-new.json'
    extend_data(
        src_path=ba_model,
        dst_path=main_data_file,
        label=ba_label
    )
    labels.append(ba_label)

    cal_draw_mae(model_paths=models,
                    labels=labels,
                    config_path='config.json',
                    save_fig=False,
                    mode='load',
                    data_path=main_data_file,
                    data_slice=100)
    
    # 运行MAE表格计算并打印程序
    extract_and_print_mae_table(data_path=main_data_file,
                                    indices=[10, 30, 100, 365, 800, 1100],
                                    table_format='grid',
                                    show_stats=True,
                                    show_rms=True,
                                    save_path="data/predicts/new/mae_by_day_table.txt")
