"""
配置文件 - config.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


@dataclass
class DataConfig:
    """数据处理相关配置"""
    # 文件路径
    model_target_dir: str = "models_reproduce/residual-mse"
    experiment_model_dir: str = "models"

    dataset_path: str = "data/eopc04_14_IAU2000.62-now.csv"
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    val_mix: bool = False
    residual_type: str = 'both'  # 'none', 'x', 'y', 'both'
    
    # Scaler相关配置
    use_scaler: bool = True
    scaler_type: str = 'minmax'  # 'minmax', 'standard', 'robust', 'none'
    scaler_after_residual: bool = False  # 是否在residual处理后应用scaler
    scaler_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.scaler_params is None:
            self.scaler_params = {}


@dataclass
class ModelConfig:
    """模型结构相关配置"""
    num_features: int = 2
    lstm0: int = 64
    lstm1: int = 64
    lstm2: int = 32
    attnhead: int = 2
    attndim: int = 64
    dropout0: float = 0.3
    dropout1: float = 0.3
    dropout2: float = 0.2


@dataclass
class TrainingConfig:
    """训练相关配置"""
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    shuffle: bool = True
    early_stop: int = 5
    loss: str = 'mae-corr'
    corr_alpha: float = 5e-4
    metrics: str = 'mae'


def get_scaler_class(scaler_type: str):
    """根据类型字符串返回scaler类"""
    scaler_map = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler,
        'none': None
    }
    
    if scaler_type not in scaler_map:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    return scaler_map[scaler_type]


# 默认配置实例
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()