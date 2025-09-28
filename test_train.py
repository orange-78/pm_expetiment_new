"""
重构后的测试主程序 - test_main_refactored.py
"""

import sys
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from config import DataConfig, ModelConfig, TrainingConfig
from model_tester import ModelTester, ModelEvaluator
from file_selector import FileSelector


class TestRunner:
    """测试运行器"""
    
    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()
        self.model_tester = ModelTester(self.data_config)
        self.file_selector = FileSelector()
        self.evaluator = ModelEvaluator()
    
    def test_single_model(self, 
                         model_path: str,
                         do_predict: List[int] = [0, 0, 1],
                         print_summary: bool = False,
                         plot_results: bool = True) -> dict:
        """测试单个模型"""
        
        print(f"\n{'='*80}")
        print(f"Testing model: {Path(model_path).name}")
        print(f"{'='*80}")
        
        # 测试模型
        results = self.model_tester.run_evaluation(
            filepath=model_path,
            do_predict=do_predict,
            print_summary=print_summary
        )
        
        # 绘制结果
        if plot_results:
            self.plot_predictions(results)
        
        return results
    
    def test_multiple_models(self, 
                           folder_paths: List[str],
                           max_depth: int = 3,
                           auto_select: bool = False) -> List[dict]:
        """测试多个模型"""
        
        if auto_select:
            # 自动选择所有文件
            file_list = self.file_selector.find_h5_files(folder_paths, max_depth)
            selected_files = [full_path for _, full_path in file_list]
        else:
            # 交互式选择
            selected_files = self.file_selector.batch_select_files(folder_paths, max_depth)
        
        if not selected_files:
            print("没有选择任何文件")
            return []
        
        results = []
        for model_path in selected_files:
            try:
                result = self.test_single_model(
                    model_path=model_path,
                    do_predict=[0, 0, 1],  # 只测试test集
                    plot_results=False  # 批量测试时不绘图
                )
                results.append(result)
                
            except Exception as e:
                print(f"测试模型 {model_path} 时出错: {e}")
                continue
        
        return results
    
    def compare_models(self, results_list: List[dict]) -> None:
        """比较多个模型的性能"""
        
        if len(results_list) < 2:
            print("至少需要两个模型才能进行比较")
            return
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        # 提取比较数据
        comparison_data = []
        for result in results_list:
            model_name = Path(result['model'].name if hasattr(result['model'], 'name') else 'Unknown').stem
            
            # 获取测试集指标
            if 'test' in result['metrics']:
                test_metrics = result['metrics']['test']
                comparison_data.append({
                    'name': model_name,
                    'overall_mae': test_metrics['overall']['mae'],
                    'overall_mse': test_metrics['overall']['mse'],
                    'overall_r2': test_metrics['overall']['r2'],
                    'feature_0_mae': test_metrics.get('feature_0', {}).get('mae', 0),
                    'feature_1_mae': test_metrics.get('feature_1', {}).get('mae', 0),
                })
        
        # 按MAE排序
        comparison_data.sort(key=lambda x: x['overall_mae'])
        
        # 打印比较表格
        print(f"{'Model':<30} {'Overall MAE':<12} {'Overall MSE':<12} {'Overall R²':<12} {'X MAE':<10} {'Y MAE':<10}")
        print("-" * 90)
        
        for data in comparison_data:
            print(f"{data['name']:<30} "
                  f"{data['overall_mae']:<12.6f} "
                  f"{data['overall_mse']:<12.6f} "
                  f"{data['overall_r2']:<12.6f} "
                  f"{data['feature_0_mae']:<10.6f} "
                  f"{data['feature_1_mae']:<10.6f}")
    
    def plot_predictions(self, results: dict, split: str = 'test', max_samples: int = 5) -> None:
        """绘制预测结果"""
        
        if split not in results['predictions'] or results['predictions'][split] is None:
            print(f"没有 {split} 集的预测结果")
            return
        
        y_true = results['ground_truth'][split]
        y_pred = results['predictions'][split]
        
        if y_true is None or y_pred is None:
            print(f"缺少 {split} 集的数据")
            return
        
        # 选择要绘制的样本
        n_samples = min(max_samples, y_true.shape[0])
        indices = np.linspace(0, y_true.shape[0]-1, n_samples, dtype=int)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # 绘制X分量
            axes[i, 0].plot(y_true[idx, :, 0], 'b-', label='True X', linewidth=2)
            axes[i, 0].plot(y_pred[idx, :, 0], 'r--', label='Pred X', linewidth=2)
            axes[i, 0].set_title(f'Sample {idx} - X Component')
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            
            # 绘制Y分量
            axes[i, 1].plot(y_true[idx, :, 1], 'b-', label='True Y', linewidth=2)
            axes[i, 1].plot(y_pred[idx, :, 1], 'r--', label='Pred Y', linewidth=2)
            axes[i, 1].set_title(f'Sample {idx} - Y Component')
            axes[i, 1].legend()
            axes[i, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_testing_loop(self, folder_paths: List[str]) -> None:
        """交互式测试循环"""
        
        while True:
            print(f"\n{'='*60}")
            print("INTERACTIVE MODEL TESTING")
            print(f"{'='*60}")
            print("1. 测试单个模型")
            print("2. 批量测试模型")
            print("3. 比较模型性能")
            print("4. 退出")
            print("-" * 60)
            
            try:
                choice = input("请选择操作 (1-4): ").strip()
                
                if choice == '1':
                    # 测试单个模型
                    model_path = self.file_selector.select_h5_file(folder_paths)
                    if model_path:
                        self.test_single_model(model_path, plot_results=True)
                
                elif choice == '2':
                    # 批量测试
                    results = self.test_multiple_models(folder_paths)
                    if results:
                        print(f"批量测试完成，共测试了 {len(results)} 个模型")
                
                elif choice == '3':
                    # 比较模型
                    print("选择要比较的模型...")
                    results = self.test_multiple_models(folder_paths)
                    if len(results) >= 2:
                        self.compare_models(results)
                    else:
                        print("需要至少选择2个模型进行比较")
                
                elif choice == '4':
                    print("退出测试程序")
                    break
                
                else:
                    print("无效的选择，请输入 1-4")
            
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"操作过程中发生错误: {e}")


def create_test_configs():
    """创建不同的测试配置"""
    
    configs = {
        'default': DataConfig(),
        
        'minmax_before_residual': DataConfig(
            use_scaler=True,
            scaler_type='minmax',
            scaler_after_residual=False,
            residual_type='both'
        ),
        
        'standard_after_residual': DataConfig(
            use_scaler=True,
            scaler_type='standard',
            scaler_after_residual=True,
            residual_type='both'
        ),
        
        'no_scaler': DataConfig(
            use_scaler=False,
            residual_type='both'
        ),
        
        'no_residual': DataConfig(
            use_scaler=True,
            scaler_type='minmax',
            residual_type='none'
        )
    }
    
    return configs


def main():
    """主函数"""
    
    # 定义模型文件夹路径
    model_folders = [
        "models_reproduce/residual-mse",
        "models",
        "models_refactored"
    ]
    
    # 获取测试配置
    configs = create_test_configs()
    
    print("Available test configurations:")
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"{i}. {name}: scaler={config.scaler_type}, residual={config.residual_type}")
    
    # 选择配置
    try:
        choice = input(f"\n请选择配置 (1-{len(configs)}) 或按回车使用默认配置: ").strip()
        if choice:
            config_names = list(configs.keys())
            selected_config = configs[config_names[int(choice)-1]]
            print(f"使用配置: {config_names[int(choice)-1]}")
        else:
            selected_config = configs['default']
            print("使用默认配置")
    except:
        selected_config = configs['default']
        print("使用默认配置")
    
    # 创建测试运行器
    test_runner = TestRunner(data_config=selected_config)
    
    # 启动交互式测试
    test_runner.interactive_testing_loop(model_folders)


def batch_test_example():
    """批量测试示例"""
    
    model_folders = ["models_reproduce/residual-mse"]
    
    # 使用不同配置测试同一个模型
    configs = create_test_configs()
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing with configuration: {config_name}")
        print(f"{'='*80}")
        
        test_runner = TestRunner(data_config=config)
        
        # 自动选择第一个模型进行测试
        file_list = test_runner.file_selector.find_h5_files(model_folders)
        if file_list:
            model_path = file_list[0][1]  # 选择第一个模型
            result = test_runner.test_single_model(
                model_path=model_path,
                plot_results=False
            )


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 或运行批量测试示例
    # batch_test_example()