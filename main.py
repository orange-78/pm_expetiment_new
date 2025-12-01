"""
重构后的主程序 - main_refactored.py
"""

import json
import pickle
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import os

from matplotlib import pyplot as plt
import numpy as np

# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from config import DataConfig, ModelConfig, TrainingConfig
from data_pipeline import DataPipeline
from model_factory import ModelFactory
from trainer import TrainingPipeline, Trainer
from model_tester import ModelTester
from data_handler import DataManager
from visualizer import plot_grid_graph, plot_pm, plot_pm_with_history
from csv_data_manager import CSVDataManager
from model_runner import ModelRunner
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, load_config
from error_visualization import calculate_mae_by_step, calculate_mae_of_dataset, plot_mae_by_step


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, 
                 data_config: Optional[DataConfig] = None,
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        
        self.data_config = data_config or DATA_CONFIG
        self.model_config = model_config or MODEL_CONFIG
        self.training_config = training_config or TRAINING_CONFIG
        
        # 初始化组件
        self.data_pipeline = DataPipeline(self.data_config)
        self.model_factory = ModelFactory()
        self.trainer = Trainer(self.training_config)
        self.training_pipeline = TrainingPipeline(
            self.data_pipeline, self.model_factory, self.trainer
        )
        self.model_tester = ModelTester(self.data_config)
    
    def single_experiment(self,
                         lookback: int,
                         steps: int,
                         model_name: str,
                         model_type: str = 'lstm_attention',
                         append_params: bool = True,
                         full_batch: bool = False) -> tuple:
        """运行单个实验"""
        
        # 构建保存名称
        if append_params:
            save_name = f"{model_name}-{lookback}_{steps}"
        else:
            save_name = model_name
        
        print(f"\n{'='*60}")
        print(f"Starting experiment: {save_name}")
        print(f"Lookback: {lookback}, Steps: {steps}")
        print(f"Model type: {model_type}")
        print(f"{'='*60}")
        
        # 运行训练
        model, history, data_info = self.training_pipeline.run_training(
            model_type=model_type,
            lookback=lookback,
            steps=steps,
            model_config=self.model_config,
            save_name=save_name,
            full_batch=full_batch
        )
        
        print(f"Experiment {save_name} completed!")
        
        return model, history, data_info
    
    def batch_experiments(self,
                         lookbacks: List[int],
                         interval: int = 30,
                         start_at: int = 0,
                         end_at: List[int] = None,
                         model_name_prefix: str = "model",
                         model_type: str = 'lstm_attention') -> List[tuple]:
        """批量实验"""
        
        if end_at is None:
            end_at = []
        
        results = []
        
        for i, lookback in enumerate(lookbacks):
            max_steps = min(end_at[i], lookback) if i < len(end_at) else lookback
            
            j = 1
            while interval * j <= max_steps:
                steps = interval * j
                
                if steps < start_at:
                    j += 1
                    continue
                
                model_name = f"{str(lookback)}_{str(steps)}/{model_name_prefix}"
                
                try:
                    result = self.single_experiment(
                        lookback=lookback,
                        steps=steps,
                        model_name=model_name,
                        model_type=model_type,
                        append_params=False
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error in experiment {model_name}: {e}")
                    continue
                
                finally:
                    j += 1
        
        return results
    
    def test_model(self, model_path: str, **kwargs):
        """测试已训练的模型"""
        return self.model_tester.run_evaluation(model_path, **kwargs)


def select_model_file(folder_paths, max_depth=3):
    """
    在多个文件夹及其子目录中查找.h5或.keras文件，按字母顺序编号并让用户选择
    
    参数:
    folder_paths: 字符串或字符串列表，文件夹路径
    max_depth: int, 最大检索深度（默认3层）
    
    返回:
    选择的.h5或.keras文件的绝对路径
    """
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]  # 单个路径转为列表
    
    all_h5_files = []
    
    for folder_path in folder_paths:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"路径不是文件夹: {folder_path}")

        base_depth = folder_path.rstrip(os.sep).count(os.sep)
        folder_name = os.path.basename(os.path.normpath(folder_path))

        # 遍历文件夹（限制深度）
        for root, dirs, files in os.walk(folder_path):
            current_depth = root.count(os.sep) - base_depth
            if current_depth >= max_depth:
                dirs[:] = []  # 不再深入
                continue
            for file in files:
                if file.endswith(".h5") or file.endswith(".keras"):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, folder_path)
                    # 在相对路径前加上顶层目录名，避免冲突
                    combined_relpath = os.path.join(folder_name, relative_path)
                    all_h5_files.append((combined_relpath, full_path))

    # 按相对路径排序
    all_h5_files.sort(key=lambda x: x[0])
    
    if not all_h5_files:
        print(f"在 {folder_paths} 及其 {max_depth} 层子目录中没有找到.h5或.keras文件")
        return None
    
    # 显示文件列表
    print(f"在 {folder_paths} 及其 {max_depth} 层子目录中找到 {len(all_h5_files)} 个.h5或.keras文件:")
    print("-" * 50)
    
    for i, (rel_path, _) in enumerate(all_h5_files, 1):
        print(f"{i:2d}. {rel_path}")
    
    print("-" * 50)
    
    # 获取用户输入
    while True:
        try:
            choice = input("请选择文件编号 (输入q退出): ").strip()
            
            if choice.lower() == 'q':
                print("用户选择退出")
                return None
            
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(all_h5_files):
                rel_path, full_path = all_h5_files[choice_num - 1]
                print(f"已选择: {rel_path}")
                return full_path
            else:
                print(f"请输入 1-{len(all_h5_files)} 之间的数字")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户中断操作")
            return None



def create_custom_config():
    """创建自定义配置的示例"""
    
    # 数据配置 - 使用不同的scaler和残差设置
    data_config = DataConfig(
        model_target_dir="models_refactored",
        dataset_path="eopc04_14_IAU2000.62-now.csv",
        train_ratio=0.75,
        val_ratio=0.15,
        residual_type='both',  # 'none', 'x', 'y', 'both'
        use_scaler=True,
        scaler_type='standard',  # 'minmax', 'standard', 'robust', 'none'
        scaler_after_residual=False,  # 在残差处理前还是后应用scaler
        scaler_params={'feature_range': (0, 1)} if 'minmax' else {}
    )
    
    # 模型配置
    model_config = ModelConfig(
        model_target_dir= "data/models_reproduce/residual-mse",
        lstm0=64,
        lstm1=64,
        lstm2=32,
        attnhead=4,  # 增加注意力头数
        attndim=32,
        dropout0=0.2,
        dropout1=0.2,
        dropout2=0.1
    )
    
    # 训练配置
    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        epochs=100,
        early_stop=10,
        loss='mae-corr',
        corr_alpha=1e-3
    )
    
    return data_config, model_config, training_config


def train_main(lookback: List[int],
               steps: int,
               model_name: str,
               use_batch: bool,
               interval: int,
               start_at: int,
               end_at: int):
    """训练主函数"""
    
    # 1. 使用默认配置
    print("Using default configuration...")
    runner_default = ExperimentRunner()

    # # 2. 使用自定义配置
    # print("Using custom configuration...")
    # data_config, model_config, training_config = create_custom_config()
    # runner_custom = ExperimentRunner(data_config, model_config, training_config)
    
    if use_batch:
        # 批量实验
        results = runner_default.batch_experiments(
            lookbacks=lookback,
            interval=interval,
            start_at=start_at,
            end_at=[lookback[i] - end_at for i in range(len(lookback))] if end_at <= 0
            else [end_at for i in range(len(lookback))],
            model_name_prefix=model_name,
            model_type=MODEL_CONFIG.model_type
        )
    else:
        # 单个实验
        if len(lookback) > 1:
            print(f"only support single lookback when not batch mode, recieved {str(len(lookback))}")
        else:
            model, history, data_info = runner_default.single_experiment(
                lookback=lookback[0],
                steps=steps,
                model_name=model_name,
                model_type=MODEL_CONFIG.model_type
            )
    
    print("Main function completed.")

def test_main(model_path: str = None, data_index: int = -1):
    """测试主函数"""
    # 1. 使用默认配置
    print("Using default configuration...")
    runner_default = ExperimentRunner()
    # 2. 测试模型
    if not model_path:
        model_path = select_model_file(runner_default.data_config.model_target_dir, max_depth=7)
        single_test = False
    else:
        single_test = True
    while model_path:
        if os.path.exists(model_path):
            print("Testing existing model...")
            test_results = runner_default.test_model(
                model_path=model_path,
                do_predict=[0, 0, 1],  # 仅预测测试集
                print_summary=True
            )
            T_test = np.concatenate([test_results['data']['raw_data'][4], test_results['ground_truth']['test']], axis=1)
            p_test = np.concatenate([test_results['data']['raw_data'][4], test_results['predictions']['test']], axis=1)
            plot_pm(T_test, p_test, data_index)
        else:
            print(f"{model_path} doesn't exist!")
        if single_test:
            break
        else:
            model_path = select_model_file(runner_default.data_config.model_target_dir, max_depth=7)
    
    print("Main function completed.")

def val_main(repo_path: str, model_name: str, data_path: str):
    """评估主函数"""
    if not Path(repo_path).exists():
        raise ValueError(f"根目录不存在: {repo_path}")
    
    # 获取基础评估结果表
    data_manager = DataManager(repo_path, excel_filename=data_path)

    # 逐模型进行评估
    config_path = f"{repo_path}/config.json"
    data_cfg, model_cfg, training_cfg = load_config(config_path)
    tester = ModelTester(data_cfg)

    for model_info in data_manager.get_existing_model_paths_with_configs(model_name):
        model_path, lookback, steps = model_info
        print(f"\n 正在评估模型: {model_path}")

        result = tester.load_and_test_model(model_path, [0, 0, 1], False)
        metrics: dict = result['metrics']['test']

        # 确认目标行
        rows = data_manager.locate_row_by_keys("lookback", "steps", lookback, steps)
        if not rows:
            print(f" 未找到 lookback={lookback}, steps={steps} 的行，跳过。")
            continue
        row_idx = rows[0]

        # 检查是否已经完整写过数据（即所有指标列都非空）
        already_complete = True
        for name, content in metrics.items():
            for metric_name in content.keys():
                col_name = f"{name}_{metric_name}"
                if col_name not in data_manager.get_all_headers():
                    already_complete = False
                    break
                val = data_manager.get_row_data(row_idx)[data_manager.header_map[col_name]-1]
                if val is None:
                    already_complete = False
                    break
            if not already_complete:
                break
        if already_complete:
            print(f" 行 lookback={lookback}, steps={steps} 已存在完整指标，跳过。")
            continue

        # 写入指标
        for name, content in metrics.items():
            for metric_name, value in content.items():
                col_name = f"{name}_{metric_name}"
                # 如果列不存在 → 新增
                if col_name not in data_manager.get_all_headers():
                    data_manager.add_empty_column(col_name)
                # 写值
                data_manager.modify_cell_by_keys(
                    "lookback", "steps", lookback, steps, 
                    col_name, float(value), limit_one=True
                )
        print(f" 写入完成: lookback={lookback}, steps={steps}")

    # 保存Excel
    data_manager.save()
    print(f"\n 已保存评估结果至 {data_manager.get_excel_path()}")

def plot_main(repo_path: str, data_path: str):
    """绘图主函数"""
    data = DataManager(repo_path, excel_filename=data_path)
    plot_grid_graph(data.get_column_data('lookback'),
                    data.get_column_data('steps'),
                    data.get_column_data('overall_pcc'),
                    title='',
                    metric_name='Corrcoef',
                    unit='',
                    scale=1.0,
                    figsize=(16, 8),
                    reverse_colorbar_num=False,
                    reverse_colorbar_color=True,
                    cmap='viridis',
                    font_size=28,
                    vrange=(0.701, 0.999))
    
def predict_main(model_path: str, csv_path: str, 
                 save_path: str = None):
    """
    主预测函数：加载模型、读取CSV最新序列、进行预测并写入结果。

    Args:
        model_path: 模型文件路径 (.keras / .h5)
        csv_path: 要预测的CSV文件路径
        train_csv_path: 可选，用于拟合Scaler的训练数据路径
        save_path: 可选，预测结果保存路径（默认覆盖原CSV）
    """

    print("=" * 60)
    print("🚀 开始执行 predict_main()")
    print("=" * 60)

    # === 1️⃣ 加载CSV数据 ===
    csv_manager = CSVDataManager(csv_path)
    csv_manager.print_summary()

    # === 2️⃣ 创建模型运行器 ===
    # ⚠️ 注意：DataConfig 必须与训练时保持一致！
    data_config, _, _ = load_config("config_base.json")

    runner = ModelRunner(model_path, data_config)

    # === 3️⃣ 获取最新lookback序列 ===
    lookback = runner.lookback
    steps = runner.steps

    START_MJD = 51544
    print(f"读取（不一定）最新 {lookback} 条记录作为输入序列...")
    # input_seq = csv_manager.read_latest_sequence(
    #     length=lookback
    # )
    input_seq = csv_manager.read_sequence_by_mjd(
        start_mjd=START_MJD - 1200,
        length=lookback
    )
    print(f"输入数据形状: {input_seq.shape}")

    # === 4️⃣ 拟合Scaler ===
    runner.fit_scaler_from_data()

    # === 5️⃣ 模型预测 ===
    print(f"正在使用模型预测未来 {steps} 步...")
    predictions = runner.predict(input_seq)
    print(f"✅ 预测完成: 结果形状 = {predictions.shape}")

    # === 6️⃣ 写入预测结果 ===
    print("正在将预测结果写入CSV...")
    # csv_manager.append_predictions_from_last(predictions, save_path=save_path)
    csv_manager.write_predictions(predictions=predictions, 
                                  start_date=START_MJD)
    print("✅ CSV写入完成")

    print("=" * 60)
    print("🎯 预测流程已结束")
    print("=" * 60)

def draw_main(csv_path: str):
    """预测数据绘制函数"""
    # ===加载CSV数据 ===
    csv_manager = CSVDataManager(csv_path)
    csv_manager.print_summary()

    # history_data = csv_manager.read_predictions_by_date_range('x_pole', 'y_pole',
    #                                                           '2023-7-9', '2025-9-15')
    history_data = csv_manager.read_predictions_by_date_range('x_pole', 'y_pole',
                                                              '1997-1-1', '1999-12-31')

    # bullitenA_data = csv_manager.read_predictions_by_date_range('a_x_pole_predict','a_y_pole_predict',
    #                                                             '2025-9-16', '2026-9-11')
    bullitenA_data = csv_manager.read_predictions_by_date_range('x_pole','y_pole',
                                                                '2000-1-1', '2003-1-4')
    
    # our_data = csv_manager.read_predictions_by_date_range('x_pole_predict','y_pole_predict',
    #                                                             '2025-9-16', '2027-5-8')
    our_data = csv_manager.read_predictions_by_date_range('x_pole_predict','y_pole_predict',
                                                                '2000-1-1', '2003-1-4')
    
    # plot_pm(bullitenA_data, our_data, start_date='2025-9-16')
    # plot_pm_with_history(history_data, bullitenA_data, our_data,
    #                      '2025-9-16',
    #                      legend_labels=('history', 'BulletinA', 'Our Model'))
    fig = plot_pm_with_history(history_data, bullitenA_data, our_data,
                         '2000-1-1',
                         legend_labels=('history', 'true data', 'Our Model'))
    
    
    # === 自动创建目录并保存 ===
    save_path: str = "figures/compare.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)

    # 返回 Figure 以便外部继续操作（如 plt.show 或再保存）

    pass

def cal_draw_mae(model_paths: str = None, 
                 labels: List[str] = None,
                 config_path: str = None,
                 save_fig: bool = True,
                 mode: str = 'run',
                 data_path: str = None):
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
        print(f"❌ 错误: 无效的模式 '{mode}', 请使用 'run' 或 'load'")
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

# 在 __main__ 中添加调用
if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='模型训练/测试脚本')
    # 添加参数
    parser.add_argument('action', help='train or test or val or plot or predict or draw or mae')
    
    # 测试参数 - 修改为接收多个值
    parser.add_argument('--path', type=str, nargs='+', help='test model path(s)', default=None)
    parser.add_argument('--index', type=int, help='test data index', default=-1)
    
    # 训练参数
    parser.add_argument('--lookback', type=int, nargs='+', help='train lookback (int or list of int)', default=[200])
    parser.add_argument('--steps', type=int, help='train steps', default=100)
    parser.add_argument('--name', type=str, help='train model name', default='model')
    parser.add_argument('--batch', action='store_true', help='train model with batch', default=False)
    parser.add_argument('--interval', type=int, help='batch steps interval', default=100)
    parser.add_argument('--startstep', type=int, help='batch steps start', default=0)
    parser.add_argument('--endstep', type=int, help='batch steps end', default=0)
    
    # 评估和绘制结果图参数
    parser.add_argument('--repopath', type=str, help='repo to evaluate', default='')
    parser.add_argument('--modelname', type=str, help='model name to scan', default='')
    parser.add_argument('--dataname', type=str, help='xlsx file name', default='evaluation')
    
    # 预测参数
    parser.add_argument('--modelpath', type=str, help='prediction model path', default='')
    parser.add_argument('--csvpath', type=str, help='csv data path', default='')
    
    # MAE绘制参数 - 修改标签参数为接收多个值
    parser.add_argument('--savefig', action='store_true', help='save mae figure', default=False)
    parser.add_argument('--labels', type=str, nargs='+', help='labels for each model in the plot', default=None)
    parser.add_argument('--maemode', type=str,  help='save to or load from data path, "load" or "run"', default='run')
    parser.add_argument('--maedatapath', type=str,  help='mae save load data path, "load" or "run"', default='data/predicts/mae_figure_data.json')
    
    # 配置参数
    parser.add_argument('--cfgpath', type=str, help='config relative full name', default=None)
    
    # 解析参数
    args = parser.parse_args()
    
    if args.action == "train":
        # 运行训练主程序
        train_main(lookback=args.lookback,
                   steps=args.steps,
                   model_name=args.name,
                   use_batch=args.batch,
                   interval=args.interval,
                   start_at=args.startstep,
                   end_at=args.endstep)
    elif args.action == "test":
        # 运行测试主程序
        test_main(model_path=args.path,
                  data_index=args.index)
    elif args.action == "val":
        # 运行模型评估程序
        val_main(repo_path=args.repopath,
                 model_name=args.modelname,
                 data_path=args.dataname)
    elif args.action == "plot":
        plot_main(repo_path=args.repopath,
                  data_path=args.dataname)
    elif args.action == "predict":
        predict_main(model_path=args.modelpath,
                     csv_path=args.csvpath)
    elif args.action == "draw":
        draw_main(csv_path=args.csvpath)
    elif args.action == "mae":
        # 运行MAE计算和绘制程序
        cal_draw_mae(model_paths=args.path,
                     labels=args.labels,
                     config_path=args.cfgpath,
                     save_fig=args.savefig,
                     mode=args.maemode,
                     data_path=args.maedatapath)

"""
最终文章用模型：
mae2
mse2
mse-corr3
mse-int3
"""
