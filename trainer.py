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
        loss_kwargs = {'alpha': self.config.alpha, 'base_loss': self.config.base_loss}
        loss_func = self.loss_functions.get_loss_function(self.config.loss, **loss_kwargs)
        
        # 获取评估指标
        metrics_kwargs = {'loss_name': self.config.loss, **loss_kwargs}
        metrics = self.loss_functions.get_metrics(self.config.metrics, **metrics_kwargs)
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss_func,
            metrics=metrics
        )
        
        return model
    
    def create_callbacks(self, checkpoint_path: str) -> list:
        """创建训练回调函数"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # 从配置中获取监控指标
        monitor_metric = getattr(self.config, "monitor", "val_loss")

        callbacks = [
            # 保存损失最佳模型
            ModelCheckpoint(
                filepath=checkpoint_path.replace('.keras', '-bestloss.keras'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # 保存指标最佳模型
            ModelCheckpoint(
                filepath=checkpoint_path.replace('.keras', '-bestmetric.keras'),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            ),
            # 保存最后一个模型
            ModelCheckpoint(
                filepath=checkpoint_path.replace('.keras', '-last.keras'),
                monitor=monitor_metric,
                save_best_only=False,
                save_weights_only=False,
                verbose=0
            )
        ]

        # 添加早停回调（当 early_stop > 0 时才启用）
        if getattr(self.config, "early_stop", 0) > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor_metric,
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
              checkpoint_path: Optional[str] = None,
              full_batch: bool = False,
              callbacks: Optional[list] = None) -> Any:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_data: 训练数据 (X_train, y_train)
            val_data: 验证数据 (X_val, y_val) 
            checkpoint_path: 模型保存路径（仅在未传 callbacks 时使用）
            full_batch: 是否使用全批次训练
            callbacks: 外部传入的回调列表（优先级更高）
            
        Returns:
            训练历史
        """
        
        X_train, y_train = train_data
        
        # 设置批次大小
        batch_size = len(X_train) if full_batch else self.config.batch_size
        
        # 回调逻辑
        if callbacks is None:
            if checkpoint_path is not None:
                callbacks = self.create_callbacks(checkpoint_path)
            else:
                callbacks = []
        
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
        # 优先使用 .keras 格式
        checkpoint_path = os.path.join(
            self.trainer.config.model_target_dir,
            f"{save_name}.keras"
        )

        # 回调：保存最优模型（完整）
        checkpoint_cb = self.trainer.create_callbacks(checkpoint_path)

        history = self.trainer.train(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            callbacks=[checkpoint_cb],
            full_batch=full_batch
        )

        # # 训练完成后再保存一份最终模型
        # model.save(checkpoint_path)

        # 5. 返回结果
        data_info = {
            'scaler': scaler,
            'raw_data': raw_data,
            'test_data': (X_test, y_test),
            'datasets': datasets
        }

        return model, history, data_info
