"""
数据处理管道 - data_pipeline.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tf_singleton import tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fractions import Fraction
from typing import Tuple, Dict, Optional, Any, Union
from config import DataConfig, get_scaler_class, parse_scaler_params


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_xy_from_csv(csv_path: str, dropna: bool = True) -> np.ndarray:
        """
        读取 CSV 并按时间顺序排序，返回 shape (N, 2) 的 numpy 数组 (x_pole, y_pole).
        """
        df = pd.read_csv(csv_path)
        
        # 排序优先级
        if 'MJD' in df.columns:
            df = df.sort_values('MJD')
        elif {'Year', 'Month', 'Day'}.issubset(df.columns):
            df = df.assign(_date=pd.to_datetime(df[['Year','Month','Day']])).sort_values('_date').drop(columns=['_date'])
            
        # 必要列检查
        if 'x_pole' not in df.columns or 'y_pole' not in df.columns:
            raise ValueError("CSV must contain 'x_pole' and 'y_pole' columns.")
            
        if dropna:
            df = df.dropna(subset=['x_pole','y_pole']).reset_index(drop=True)
            
        arr = df[['x_pole','y_pole']].astype(np.float32).values
        return arr


class DataSplitter:
    """数据分割器"""
    
    @staticmethod
    def time_split(data: np.ndarray, train_ratio: float=0.7, val_ratio: float=0.15
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按时间顺序分割数据"""
        N = len(data)
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and train_ratio + val_ratio <= 1):
            raise ValueError("Invalid ratios. Require 0<=train_ratio<=1, 0<=val_ratio<=1 and train_ratio+val_ratio<=1.")
            
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        
        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]
        
        return train, val, test
    
    @staticmethod
    def split_array_alternating_nd(A: np.ndarray, m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """交替分配数组元素"""
        if not isinstance(A, np.ndarray):
            raise TypeError("输入A必须是numpy数组")
        if m <= 0 or n <= 0:
            raise ValueError("m和n必须是正整数")
        
        total_elements = A.shape[0]
        B_indices = []
        C_indices = []
        
        start = 0
        assign_to_B = True
        
        while start < total_elements:
            if assign_to_B:
                end = min(start + m, total_elements)
                B_indices.extend(range(start, end))
                assign_to_B = False
            else:
                end = min(start + n, total_elements)
                C_indices.extend(range(start, end))
                assign_to_B = True
            
            start = end
        
        B = A[B_indices, ...] if B_indices else np.array([], dtype=A.dtype)
        C = A[C_indices, ...] if C_indices else np.array([], dtype=A.dtype)
        
        return B, C
    
    @staticmethod
    def find_smallest_ratio_integers(a: float, b: float, max_denominator: int=50) -> Tuple[int, int]:
        """找到最小整数比例"""
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("a和b必须在0-1之间")
        if a == 0 and b == 0:
            return 0, 0
        
        if b == 0:
            ratio = float('inf') if a > 0 else 0
        else:
            ratio = a / b
        
        try:
            frac = Fraction(ratio).limit_denominator(max_denominator)
            A = frac.numerator
            B = frac.denominator
        except:
            A = round(a * 10)
            B = round(b * 10)
            if A == 0 and B == 0:
                A, B = 1, 1
        
        return A, B


class DataScaler:
    """数据缩放器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = None
        
        if config.use_scaler and config.scaler_type != 'none':
            scaler_class = get_scaler_class(config.scaler_type)
            if scaler_class is not None:
                scaler_params = parse_scaler_params(config.scaler_params)
                self.scaler = scaler_class(**scaler_params)
    
    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """拟合并转换训练数据"""
        if self.scaler is None:
            return train_data.copy()
        
        self.scaler.fit(train_data)
        return self.scaler.transform(train_data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """转换数据"""
        if self.scaler is None:
            return data.copy()
        
        if len(data) == 0:
            return data.copy()
            
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆转换数据"""
        if self.scaler is None or data is None or data.size == 0:
            return data
        
        # 处理tensorflow tensor
        if isinstance(data, tf.Tensor):
            data = data.numpy()
            
        original_shape = data.shape
        if len(original_shape) > 2:
            # 展平为2D进行逆转换
            flat_data = data.reshape(-1, original_shape[-1])
            inv_flat = self.scaler.inverse_transform(flat_data)
            return inv_flat.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)


class ResidualProcessor:
    """残差处理器"""
    
    @staticmethod
    def make_residual_x(X: np.ndarray) -> np.ndarray:
        """计算X的残差"""
        X = tf.constant(X)
        dX = X[:, 1:, :] - X[:, :-1, :]
        zero_pad = tf.zeros_like(X[:, :1, :])
        return tf.concat([zero_pad, dX], axis=1).numpy()
    
    @staticmethod
    def make_residual_y(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算Y的残差"""
        X = tf.constant(X)
        Y = tf.constant(Y)
        
        first_diff = tf.expand_dims(Y[:, 0, :] - X[:, -1, :], axis=1)
        dY = Y[:, 1:, :] - Y[:, :-1, :]
        
        return tf.concat([first_diff, dY], axis=1).numpy()
    
    @staticmethod
    def reconstruct_residual_x_batch(X_original: np.ndarray, X_residual: np.ndarray) -> np.ndarray:
        """批量重构X残差"""
        X_original = tf.constant(X_original)
        X_residual = tf.constant(X_residual)
        
        init = X_original[:, 0:1, :]
        cumsum = tf.cumsum(X_residual, axis=1)
        
        return (init + cumsum).numpy()
    
    @staticmethod
    def reconstruct_residual_y_batch(X_original: np.ndarray, y_residual: np.ndarray) -> np.ndarray:
        """批量重构Y残差"""
        X_original = tf.constant(X_original)
        y_residual = tf.constant(y_residual)
        
        init = X_original[:, -1:, :]
        cumsum = tf.cumsum(y_residual, axis=1)
        
        return (init + cumsum).numpy()


class WindowGenerator:
    """滑动窗口生成器"""
    
    @staticmethod
    def create_sliding_windows(data: np.ndarray, lookback: int, steps: int
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口"""
        N = len(data)
        L = lookback
        S = steps
        
        if N < L + S:
            return np.zeros((0, L, 2), dtype=np.float32), np.zeros((0, S, 2), dtype=np.float32)
        
        n_samples = N - L - S + 1
        X = np.empty((n_samples, L, 2), dtype=np.float32)
        y = np.empty((n_samples, S, 2), dtype=np.float32)
        
        for i in range(n_samples):
            X[i] = data[i:i+L]
            y[i] = data[i+L:i+L+S]
            
        return X, y


class DataPipeline:
    """完整的数据处理管道"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.loader = DataLoader()
        self.splitter = DataSplitter()
        self.scaler = DataScaler(config)
        self.residual_processor = ResidualProcessor()
        self.window_generator = WindowGenerator()
        
        # 存储原始分割数据，用于逆变换
        self.raw_splits = None
        self.raw_windows = None
        self.raw_windows_scaled = None
        
    def prepare_datasets(self, csv_path: str, lookback: int, steps: int
                        ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        完整的数据准备流程
        返回: (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, raw_data
        """
        
        # 1. 加载数据
        data = self.loader.load_xy_from_csv(csv_path)
        
        # 2. 分割数据
        if self.config.val_mix:
            train_val_raw, _, test_raw = self.splitter.time_split(
                data, self.config.train_ratio + self.config.val_ratio, 0
            )
            portion_train, portion_val = self.splitter.find_smallest_ratio_integers(
                self.config.train_ratio, self.config.val_ratio
            )
            train_raw, val_raw = self.splitter.split_array_alternating_nd(
                train_val_raw, portion_train, portion_val
            )
        else:
            train_raw, val_raw, test_raw = self.splitter.time_split(
                data, self.config.train_ratio, self.config.val_ratio
            )
        
        self.raw_splits = (train_raw, val_raw, test_raw)
        
        # 3. 第一次缩放（如果不是在残差后缩放）
        if self.config.use_scaler and not self.config.scaler_after_residual:
            train_s = self.scaler.fit_transform(train_raw)
            val_s = self.scaler.transform(val_raw)
            test_s = self.scaler.transform(test_raw)
        else:
            train_s, val_s, test_s = train_raw.copy(), val_raw.copy(), test_raw.copy()
        
        # 4. 生成滑动窗口
        X_train_raw, y_train_raw = self.window_generator.create_sliding_windows(train_s, lookback, steps)
        X_val_raw, y_val_raw = self.window_generator.create_sliding_windows(val_s, lookback, steps)
        X_test_raw, y_test_raw = self.window_generator.create_sliding_windows(test_s, lookback, steps)
        
        self.raw_windows_scaled = (X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)
        if self.config.use_scaler and not self.config.scaler_after_residual:
            self.raw_windows = (self.scaler.inverse_transform(X_train_raw),
                                self.scaler.inverse_transform(y_train_raw),
                                self.scaler.inverse_transform(X_val_raw),
                                self.scaler.inverse_transform(y_val_raw),
                                self.scaler.inverse_transform(X_test_raw),
                                self.scaler.inverse_transform(y_test_raw))
        else:
            self.raw_windows = self.raw_windows_scaled
        
        # 5. 残差处理
        X_train, y_train = self._apply_residual_transform(X_train_raw, y_train_raw)
        X_val, y_val = self._apply_residual_transform(X_val_raw, y_val_raw)
        X_test, y_test = self._apply_residual_transform(X_test_raw, y_test_raw)
        
        # 6. 第二次缩放（如果在残差后缩放）
        if self.config.use_scaler and self.config.scaler_after_residual:
            # 需要重新初始化scaler并拟合
            self._fit_scaler_after_residual(X_train, y_train)
            
            X_train = self._transform_after_residual(X_train)
            X_val = self._transform_after_residual(X_val)
            X_test = self._transform_after_residual(X_test)
            
            y_train = self._transform_after_residual(y_train)
            y_val = self._transform_after_residual(y_val)
            y_test = self._transform_after_residual(y_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), self.scaler, self.raw_windows
    
    def _apply_residual_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用残差变换"""
        if self.config.residual_type == 'x':
            X_out = self.residual_processor.make_residual_x(X)
            y_out = y
        elif self.config.residual_type == 'y':
            X_out = X
            y_out = self.residual_processor.make_residual_y(X, y)
        elif self.config.residual_type == 'both':
            X_out = self.residual_processor.make_residual_x(X)
            y_out = self.residual_processor.make_residual_y(X, y)
        else:  # 'none'
            X_out = X
            y_out = y
            
        return X_out, y_out
    
    def _fit_scaler_after_residual(self, X_train: np.ndarray, y_train: np.ndarray):
        """在残差处理后拟合scaler"""
        # 将训练数据合并进行拟合
        train_combined = np.concatenate([
            X_train.reshape(-1, X_train.shape[-1]),
            y_train.reshape(-1, y_train.shape[-1])
        ], axis=0)
        
        if self.scaler.scaler is not None:
            self.scaler.scaler.fit(train_combined)
    
    def _transform_after_residual(self, data: np.ndarray) -> np.ndarray:
        """在残差处理后转换数据"""
        if self.scaler.scaler is None:
            return data
            
        original_shape = data.shape
        flat_data = data.reshape(-1, original_shape[-1])
        transformed = self.scaler.scaler.transform(flat_data)
        return transformed.reshape(original_shape)
    
    def reconstruct_predictions(self, predictions: Tuple[Optional[np.ndarray], ...], 
                               residual_type: Optional[str] = None) -> Tuple[Optional[np.ndarray], ...]:
        """重构预测结果（处理残差和逆缩放）"""
        if residual_type is None:
            residual_type = self.config.residual_type
            
        o_train, o_val, o_test = predictions
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = self.raw_windows_scaled
        
        # 逆缩放（如果在残差后应用了缩放）
        if self.config.use_scaler and self.config.scaler_after_residual:
            o_train = self.scaler.inverse_transform(o_train) if o_train is not None else None
            o_val = self.scaler.inverse_transform(o_val) if o_val is not None else None
            o_test = self.scaler.inverse_transform(o_test) if o_test is not None else None
        
        # 残差重构
        if residual_type in ['y', 'both']:
            o_train = self.residual_processor.reconstruct_residual_y_batch(X_train_raw, o_train) if o_train is not None else None
            o_val = self.residual_processor.reconstruct_residual_y_batch(X_val_raw, o_val) if o_val is not None else None
            o_test = self.residual_processor.reconstruct_residual_y_batch(X_test_raw, o_test) if o_test is not None else None
        
        # 逆缩放（如果在残差前应用了缩放）
        if self.config.use_scaler and not self.config.scaler_after_residual:
            o_train = self.scaler.inverse_transform(o_train) if o_train is not None else None
            o_val = self.scaler.inverse_transform(o_val) if o_val is not None else None
            o_test = self.scaler.inverse_transform(o_test) if o_test is not None else None
        
        return o_train, o_val, o_test