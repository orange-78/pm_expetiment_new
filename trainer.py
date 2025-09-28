"""
训练器模块 - trainer.py
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.models import Model # type: ignore

from tf_singleton import tf
from config import TrainingConfig
from loss_functions import LossFunctions
from data_pipeline import DataPipeline
from model_factory import ModelFactory

class Trainer:
    """模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.loss_functions = LossFunctions()
        
    def setup_model(self, model: Model) -> Model:
        """配置模型（编译）"""
        # 获取损失函数
        loss_kwargs = {'alpha': self.config.corr_alpha} if 'corr' in self.config.loss else {}
        loss_func = self.loss_functions.get_loss_function(self.config.loss, **loss_kwargs)
        
        # 获取评估指标
        metrics_kwargs = {'loss_name': self.config.loss, **loss_kwargs}
        metrics = self.loss_functions.get_metrics(self.config.metrics, **metrics_kwargs)
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss_func,
            metrics=[metrics]
        )
        
        return model
    
    def create_callbacks(self, checkpoint_path: str) -> list:
        """创建训练回调函数"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        callbacks = [
            # 保存最佳模型
            ModelCheckpoint(
                filepath=checkpoint_path.replace('.h5', '-best.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # 保存最后一个模型
            ModelCheckpoint(
                filepath=checkpoint_path.replace('.h5', '-last.h5'),
                monitor='val_loss',
                save_best_only=False,
                save_weights_only=False,
                verbose=0
            )
        ]
        
        # 添加早停回调
        if self.config.early_stop > 0:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stop,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, 
              model: Model,
              train_data: Tuple[tf.Tensor, tf.Tensor],
              val_data: Tuple[tf.Tensor, tf.Tensor],
              checkpoint_path: str,
              full_batch: bool = False) -> Any:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_data: 训练数据 (X_train, y_train)
            val_data: 验证数据 (X_val, y_val) 
            checkpoint_path: 模型保存路径
            full_batch: 是否使用全批次训练
            
        Returns:
            训练历史
        """
        
        X_train, y_train = train_data
        
        # 设置批次大小
        batch_size = len(X_train) if full_batch else self.config.batch_size
        
        # 创建回调函数
        callbacks = self.create_callbacks(checkpoint_path)
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=val_data,
            batch_size=batch_size,
            epochs=self.config.epochs,
            shuffle=self.config.shuffle,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


class TrainingPipeline:
    """完整的训练流程"""
    
    def __init__(self, data_pipeline, model_factory, trainer):
        self.data_pipeline: DataPipeline = data_pipeline
        self.model_factory: ModelFactory = model_factory
        self.trainer: Trainer = trainer
    
    def run_training(self,
                    model_type: str,
                    lookback: int,
                    steps: int,
                    model_config,
                    save_name: str,
                    full_batch: bool = False) -> Tuple[Model, Any, Any]:
        """
        运行完整的训练流程
        
        Args:
            model_type: 模型类型
            lookback: 回望窗口大小
            steps: 预测步数
            model_config: 模型配置
            save_name: 保存文件名
            full_batch: 是否全批次训练
            
        Returns:
            (trained_model, history, data_info)
        """
        
        # 1. 准备数据
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, raw_data = self.data_pipeline.prepare_datasets(
            self.data_pipeline.config.dataset_path, lookback, steps
        )
        datasets = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        
        print(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # 2. 创建模型
        model = self.model_factory.get_model(model_type, lookback, steps, model_config)
        print(f"Model created: {model_type}")
        
        # 3. 配置模型
        model = self.trainer.setup_model(model)
        
        # 4. 训练模型
        checkpoint_path = os.path.join(
            self.trainer.config.model_target_dir, 
            f"{save_name}.h5"
        )
        
        history = self.trainer.train(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            checkpoint_path=checkpoint_path,
            full_batch=full_batch
        )
        
        # 5. 返回结果
        data_info = {
            'scaler': scaler,
            'raw_data': raw_data,
            'test_data': (X_test, y_test),
            'datasets': datasets
        }
        
        return model, history, data_info