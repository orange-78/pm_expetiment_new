"""
重构后的主程序 - main_refactored.py
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import os

import numpy as np

# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from config import DataConfig, ModelConfig, TrainingConfig
from data_pipeline import DataPipeline
from model_factory import ModelFactory
from trainer import TrainingPipeline, Trainer
from model_tester import ModelTester
from visualizer import plot_pm
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG


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
            model_name_prefix=model_name
        )
    else:
        # 单个实验
        if len(lookback) > 1:
            print(f"only support single lookback when not batch mode, recieved {str(len(lookback))}")
        else:
            model, history, data_info = runner_default.single_experiment(
                lookback=lookback[0],
                steps=steps,
                model_name=model_name
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


def demo_different_scalers():
    """演示不同scaler的使用"""
    
    base_config = {
        'dataset_path': "eopc04_14_IAU2000.62-now.csv",
        'train_ratio': 0.75,
        'val_ratio': 0.15,
        'residual_type': 'both'
    }
    
    # 测试不同的scaler配置
    scaler_configs = [
        # MinMax scaler
        {'use_scaler': True, 'scaler_type': 'minmax', 'scaler_after_residual': False,
         'scaler_params': {'feature_range': (0, 1)}},
        
        # Standard scaler
        {'use_scaler': True, 'scaler_type': 'standard', 'scaler_after_residual': False,
         'scaler_params': {}},
        
        # Robust scaler
        {'use_scaler': True, 'scaler_type': 'robust', 'scaler_after_residual': False,
         'scaler_params': {}},
        
        # Scaler after residual
        {'use_scaler': True, 'scaler_type': 'minmax', 'scaler_after_residual': True,
         'scaler_params': {'feature_range': (-1, 1)}},
        
        # No scaler
        {'use_scaler': False, 'scaler_type': 'none', 'scaler_after_residual': False,
         'scaler_params': {}}
    ]
    
    for i, scaler_config in enumerate(scaler_configs):
        print(f"\n{'='*50}")
        print(f"Testing scaler configuration {i+1}")
        print(f"Config: {scaler_config}")
        print(f"{'='*50}")
        
        # 创建数据配置
        data_config = DataConfig(**{**base_config, **scaler_config})
        
        # 创建实验运行器
        runner = ExperimentRunner(data_config=data_config)
        
        # 运行小规模实验
        try:
            model, history, data_info = runner.single_experiment(
                lookback=100,
                steps=30,
                model_name=f"scaler_test_{i+1}",
                model_type='simple_lstm'  # 使用简单模型进行快速测试
            )
            print(f"Scaler config {i+1} completed successfully!")
            
        except Exception as e:
            print(f"Error with scaler config {i+1}: {e}")


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='模型训练/测试脚本')
    # 添加参数
    parser.add_argument('action', help='train or test')
    # 测试参数
    parser.add_argument('--path', type=str, help='test model path', default=None)
    parser.add_argument('--index', type=int, help='test data index', default=-1)
    # 训练参数
    parser.add_argument('--lookback', type=int, nargs='+', help='train lookback (int or list of int)', default=[200])
    parser.add_argument('--steps', type=int, help='train steps', default=100)
    parser.add_argument('--name', type=str, help='train model name', default='model')
    parser.add_argument('--batch', action='store_true', help='train model with batch', default=False)
    parser.add_argument('--interval', type=int, help='batch steps interval', default=100)
    parser.add_argument('--startstep', type=int, help='batch steps start', default=0)
    parser.add_argument('--endstep', type=int, help='batch steps end', default=0)
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
    
    # 或者运行scaler演示
    # demo_different_scalers()