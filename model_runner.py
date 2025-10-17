"""
模型运行器 - model_runner.py
用于加载模型并进行实时预测
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from pathlib import Path

from tf_singleton import tf
from data_pipeline import DataPipeline, DataScaler, ResidualProcessor
from config import DataConfig
from model_tester import ModelLoader


class ModelRunner:
    """模型运行器 - 用于加载模型并进行实时预测"""
    
    def __init__(self, 
                 model_path: str, 
                 data_config: DataConfig,
                 custom_objects: dict = None):
        """
        初始化模型运行器
        
        Args:
            model_path: 模型文件路径
            data_config: 数据配置对象
            custom_objects: 自定义对象字典（用于加载模型）
        """
        self.model_path = model_path
        self.data_config = data_config
        self.custom_objects = custom_objects
        
        # 加载模型
        self.model = ModelLoader.load_model(model_path, custom_objects)
        
        # 提取模型参数
        self.lookback, self.steps = ModelLoader.extract_model_params(self.model)
        print(f"✅ 模型加载成功: lookback={self.lookback}, steps={self.steps}")
        
        # 初始化数据处理组件
        self.data_scaler = DataScaler(data_config)
        self.residual_processor = ResidualProcessor()
        
        # 存储scaler状态（需要从训练数据拟合）
        self.is_scaler_fitted = False
    
    def fit_scaler_from_data(self):
        """
        从训练数据拟合scaler
        
        Args:
            csv_path: 训练数据CSV路径
        """
        pipeline = DataPipeline(self.data_config)
        _, _, _, fitted_scaler, _ = pipeline.prepare_datasets(
            pipeline.config.dataset_path, self.lookback, self.steps
        )
        self.data_scaler = fitted_scaler
        self.is_scaler_fitted = True
        print("✅ Scaler已从训练数据拟合")
    
    def preprocess_input(self, 
                        input_data: np.ndarray,
                        apply_residual: bool = True) -> np.ndarray:
        """
        预处理输入数据
        
        Args:
            input_data: 输入数据，shape为 (lookback, 2) 或 (batch_size, lookback, 2)
            apply_scaling: 是否应用缩放
            apply_residual: 是否应用残差变换
            
        Returns:
            预处理后的数据
        """
        # 确保是3D数组
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)
        
        # 检查形状
        if input_data.shape[1] != self.lookback or input_data.shape[2] != 2:
            raise ValueError(
                f"输入数据形状应为 (batch_size, {self.lookback}, 2), "
                f"实际为 {input_data.shape}"
            )
        
        processed_data = input_data.copy()
        
        # 1. 应用缩放（如果在残差前）
        if (self.data_config.use_scaler and 
            not self.data_config.scaler_after_residual):
            if not self.is_scaler_fitted:
                print("⚠️ Scaler未拟合，跳过缩放步骤")
            else:
                original_shape = processed_data.shape
                flat_data = processed_data.reshape(-1, original_shape[-1])
                scaled = self.data_scaler.transform(flat_data)
                processed_data = scaled.reshape(original_shape)
        
        # 2. 应用残差变换
        if apply_residual and self.data_config.residual_type in ['x', 'both']:
            processed_data = self.residual_processor.make_residual_x(processed_data)
        
        # 3. 应用缩放（如果在残差后）
        if (self.data_config.use_scaler and 
            self.data_config.scaler_after_residual):
            if not self.is_scaler_fitted:
                print("⚠️ Scaler未拟合，跳过缩放步骤")
            else:
                original_shape = processed_data.shape
                flat_data = processed_data.reshape(-1, original_shape[-1])
                scaled = self.data_scaler.scaler.transform(flat_data)
                processed_data = scaled.reshape(original_shape)
        
        return processed_data
    
    def postprocess_output(self,
                          output_data: np.ndarray,
                          original_input: np.ndarray,
                          apply_residual: bool = True) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            output_data: 模型输出，shape为 (batch_size, steps, 2)
            original_input: 原始输入数据，用于残差重构
            apply_scaling: 是否应用逆缩放
            apply_residual: 是否重构残差
            
        Returns:
            后处理后的数据
        """
        processed_output = output_data.copy()
        
        # 1. 逆缩放（如果在残差后应用了缩放）
        if (self.data_config.use_scaler and 
            self.data_config.scaler_after_residual):
            if self.is_scaler_fitted:
                processed_output = self.data_scaler.inverse_transform(processed_output)
        
        # 2. 残差重构
        if apply_residual and self.data_config.residual_type in ['y', 'both']:
            # 需要原始输入来重构
            if original_input.ndim == 2:
                original_input = np.expand_dims(original_input, axis=0)
            processed_output = self.residual_processor.reconstruct_residual_y_batch(
                original_input, processed_output
            )
        
        # 3. 逆缩放（如果在残差前应用了缩放）
        if (self.data_config.use_scaler and 
            not self.data_config.scaler_after_residual):
            if self.is_scaler_fitted:
                processed_output = self.data_scaler.inverse_transform(processed_output)
        
        return processed_output
    
    def predict(self, 
                input_data: np.ndarray,
                preprocess: bool = True,
                postprocess: bool = True,
                return_raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        使用模型进行预测
        
        Args:
            input_data: 输入数据，shape为 (lookback, 2) 或 (batch_size, lookback, 2)
            preprocess: 是否预处理输入
            postprocess: 是否后处理输出
            return_raw: 是否同时返回原始模型输出
            
        Returns:
            预测结果，shape为 (steps, 2) 或 (batch_size, steps, 2)
            如果return_raw=True，返回 (processed_output, raw_output)
        """
        # 保存原始输入（用于后处理）
        original_input = input_data.copy()
        squeeze_output = False
        
        # 确保是3D数组
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)
            squeeze_output = True
        
        # 预处理
        if preprocess:
            processed_input = self.preprocess_input(input_data)
        else:
            processed_input = input_data
        
        # 模型预测
        raw_output = self.model.predict(processed_input, verbose=0)
        
        # 后处理
        if postprocess:
            final_output = self.postprocess_output(raw_output, input_data)
        else:
            final_output = raw_output
        
        # 如果输入是2D，输出也压缩为2D
        if squeeze_output:
            final_output = np.squeeze(final_output, axis=0)
            raw_output = np.squeeze(raw_output, axis=0)
        
        if return_raw:
            return final_output, raw_output
        else:
            return final_output
    
    def predict_sequence(self,
                        initial_data: np.ndarray,
                        n_predictions: int,
                        use_true_history: bool = False,
                        true_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        滚动预测多步
        
        Args:
            initial_data: 初始数据，shape为 (lookback, 2)
            n_predictions: 预测步数
            use_true_history: 是否使用真实历史数据（用于测试）
            true_sequence: 真实序列数据，如果use_true_history=True需要提供
            
        Returns:
            预测序列，shape为 (n_predictions, 2)
        """
        if initial_data.shape != (self.lookback, 2):
            raise ValueError(f"初始数据形状应为 ({self.lookback}, 2)")
        
        predictions = []
        current_window = initial_data.copy()
        
        for i in range(n_predictions):
            # 预测下一步
            next_pred = self.predict(current_window, preprocess=True, postprocess=True)
            
            # 只取第一步预测
            if next_pred.ndim == 2:
                next_step = next_pred[0]  # shape: (2,)
            else:
                next_step = next_pred[0, 0]
            
            predictions.append(next_step)
            
            # 更新窗口
            if use_true_history and true_sequence is not None:
                # 使用真实数据
                if i < len(true_sequence):
                    next_step = true_sequence[i]
            
            current_window = np.vstack([current_window[1:], next_step])
        
        return np.array(predictions)
    
    def batch_predict(self, 
                     input_list: List[np.ndarray],
                     preprocess: bool = True,
                     postprocess: bool = True) -> List[np.ndarray]:
        """
        批量预测多个输入
        
        Args:
            input_list: 输入数据列表，每个元素shape为 (lookback, 2)
            preprocess: 是否预处理
            postprocess: 是否后处理
            
        Returns:
            预测结果列表
        """
        # 堆叠为批次
        batch_input = np.stack(input_list, axis=0)
        
        # 批量预测
        batch_output = self.predict(
            batch_input, 
            preprocess=preprocess, 
            postprocess=postprocess
        )
        
        # 分解为列表
        return [batch_output[i] for i in range(len(input_list))]
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'lookback': self.lookback,
            'steps': self.steps,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'data_config': {
                'residual_type': self.data_config.residual_type,
                'use_scaler': self.data_config.use_scaler,
                'scaler_type': self.data_config.scaler_type,
                'scaler_after_residual': self.data_config.scaler_after_residual
            },
            'is_scaler_fitted': self.is_scaler_fitted
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (f"ModelRunner(\n"
                f"  model_path='{info['model_path']}',\n"
                f"  lookback={info['lookback']},\n"
                f"  steps={info['steps']},\n"
                f"  scaler_fitted={info['is_scaler_fitted']}\n"
                f")")


# 使用示例
if __name__ == "__main__":
    # 配置
    config = DataConfig(
        dataset_path="data/train.csv",
        residual_type="y",
        use_scaler=True,
        scaler_type="minmax",
        scaler_after_residual=False
    )
    
    # 创建运行器
    runner = ModelRunner(
        model_path="models/best_model.keras",
        data_config=config
    )
    
    # 从训练数据拟合scaler
    runner.fit_scaler_from_data("data/train.csv")
    
    # 准备输入数据（示例：随机数据）
    input_data = np.random.randn(runner.lookback, 2).astype(np.float32)
    
    # 单次预测
    prediction = runner.predict(input_data)
    print(f"预测结果形状: {prediction.shape}")
    print(f"预测结果:\n{prediction}")
    
    # 滚动预测
    sequence_pred = runner.predict_sequence(input_data, n_predictions=10)
    print(f"\n滚动预测形状: {sequence_pred.shape}")
    
    # 批量预测
    batch_inputs = [np.random.randn(runner.lookback, 2).astype(np.float32) 
                   for _ in range(5)]
    batch_predictions = runner.batch_predict(batch_inputs)
    print(f"\n批量预测数量: {len(batch_predictions)}")
    
    # 显示模型信息
    print(f"\n{runner}")