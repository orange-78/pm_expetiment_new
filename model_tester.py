"""
模型测试器 - model_tester.py
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tf_singleton import tf
from data_pipeline import DataPipeline
from config import DataConfig


class ModelLoader:
    """模型加载器"""
    
    @staticmethod
    def load_model(filepath: str, custom_objects: dict = None):
        """加载保存的模型（兼容 .keras 和 .h5）"""
        from tensorflow import keras

        # 自动判断格式
        if filepath.endswith(".h5"):
            print("⚠️ 检测到 .h5 格式，可能丢失部分 compile 信息，推荐改用 .keras。")

        try:
            # 优先带 compile 加载
            model = keras.models.load_model(filepath, compile=True, custom_objects=custom_objects)
            if model.optimizer is not None and model.loss is not None:
                print("✅ 模型已包含 optimizer 和 loss，使用保存时配置。")
                return model
        except Exception as e:
            print(f"加载时未检测到完整编译信息，错误信息: {e}")
        
        # 如果上面失败，则手动 compile
        print("⚠️ 模型未包含 optimizer/loss 信息，使用默认配置重新编译。")
        model = keras.models.load_model(filepath, compile=False, custom_objects=custom_objects)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mae",
            metrics=["mae"]
        )
        return model
    
    @staticmethod
    def extract_model_params(model) -> Tuple[int, int]:
        """从模型结构提取参数"""
        lookback = model.input_shape[1]
        steps = model.output_shape[1]
        return lookback, steps


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        tolerance: Optional[float] = None) -> Dict[str, float]:
        """计算评估指标"""
        # 确保数据形状一致
        if y_true.shape != y_pred.shape:
            if y_true.size == y_pred.size:
                y_true = y_true.reshape(-1)
                y_pred = y_pred.reshape(-1)
            else:
                raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # 展平为1D进行计算
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        # Pearson Correlation Coefficient
        if np.std(y_true_flat) > 0 and np.std(y_pred_flat) > 0:
            pcc = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        else:
            pcc = np.nan  # 避免除零错误

        # 容许范围内比例
        tolerance_ratio = None
        if tolerance is not None:
            diffs = np.abs(y_true_flat - y_pred_flat)
            tolerance_ratio = np.mean(diffs <= tolerance)

        result = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'pcc': pcc
        }
        if tolerance is not None:
            result[f'within_tol'] = tolerance_ratio
        
        return result
    
    @staticmethod
    def evaluate_by_features(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """按特征分别评估"""
        results = {}
        
        # 整体评估
        results['overall'] = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        # 按特征评估（假设最后一个维度是特征）
        if y_true.ndim >= 2:
            for feat_idx in range(y_true.shape[-1]):
                feat_true = y_true[..., feat_idx]
                feat_pred = y_pred[..., feat_idx]
                results[f'feature_{feat_idx}'] = ModelEvaluator.compute_metrics(feat_true, feat_pred, 0.063)
        
        return results
    
    @staticmethod
    def print_metrics(metrics_dict: Dict[str, Dict[str, float]]):
        """打印评估指标"""
        for category, metrics in metrics_dict.items():
            print(f"\n{category.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name.upper()}: {value:.6f}")


class ModelTester:
    """模型测试器"""
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.data_pipeline = DataPipeline(data_config)
        self.model_loader = ModelLoader()
        self.evaluator = ModelEvaluator()
    
    def predict_sequences(self, model, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测"""
        return model.predict(X, verbose=0)
    
    def load_and_test_model(self, 
                           filepath: str,
                           do_predict: List[int] = [1, 1, 1],
                           print_summary: bool = False) -> Dict[str, Any]:
        """
        加载并测试模型
        
        Args:
            filepath: 模型文件路径
            do_predict: [train, val, test] 是否对各数据集进行预测
            print_summary: 是否打印模型结构
            
        Returns:
            包含所有测试结果的字典
        """
        
        # 1. 加载模型
        model = self.model_loader.load_model(filepath)
        if print_summary:
            model.summary()
        
        # 2. 提取模型参数
        lookback, steps = self.model_loader.extract_model_params(model)
        print(f"Detected lookback={lookback}, steps={steps}")
        
        # 3. 准备数据（使用相同的配置）
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, raw_data = self.data_pipeline.prepare_datasets(
            self.data_config.dataset_path, lookback, steps
        )
        datasets = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        
        # 4. 进行预测
        predictions = {}
        if do_predict[0]:
            predictions['train'] = self.predict_sequences(model, X_train)
        if do_predict[1]:
            predictions['val'] = self.predict_sequences(model, X_val)
        if do_predict[2]:
            predictions['test'] = self.predict_sequences(model, X_test)
        
        # 5. 重构预测结果（处理残差和逆缩放）
        pred_tuple = (
            predictions.get('train'),
            predictions.get('val'),
            predictions.get('test')
        )
        
        reconstructed_preds = self.data_pipeline.reconstruct_predictions(pred_tuple)
        o_train_y, o_val_y, o_test_y = reconstructed_preds
        
        # 6. 重构真实值
        true_tuple = (y_train if do_predict[0] else None,
                     y_val if do_predict[1] else None, 
                     y_test if do_predict[2] else None)
        
        reconstructed_true = self.data_pipeline.reconstruct_predictions(true_tuple)
        y_train_y, y_val_y, y_test_y = reconstructed_true
        
        # 7. 计算评估指标
        results = {
            'model': model,
            'config': {
                'lookback': lookback,
                'steps': steps,
                'data_config': self.data_config
            },
            'data': {
                'raw_data': raw_data,
                'datasets': datasets,
                'scaler': scaler
            },
            'predictions': {
                'train': o_train_y,
                'val': o_val_y, 
                'test': o_test_y
            },
            'ground_truth': {
                'train': y_train_y,
                'val': y_val_y,
                'test': y_test_y
            },
            'metrics': {}
        }
        
        # 计算各数据集的指标
        for split in ['train', 'val', 'test']:
            idx = ['train', 'val', 'test'].index(split)
            if do_predict[idx] and results['predictions'][split] is not None:
                y_true = results['ground_truth'][split]
                y_pred = results['predictions'][split]
                results['metrics'][split] = self.evaluator.evaluate_by_features(y_true, y_pred)
        
        return results
    
    def run_evaluation(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """运行完整评估并打印结果"""
        results = self.load_and_test_model(filepath, **kwargs)
        
        # 打印评估结果
        for split, metrics in results['metrics'].items():
            print(f"\n{'='*50}")
            print(f"EVALUATION RESULTS - {split.upper()}")
            print(f"{'='*50}")
            self.evaluator.print_metrics(metrics)
        
        return results