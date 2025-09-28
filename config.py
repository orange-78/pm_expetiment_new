"""
配置文件 - config.py
"""

import json5 as json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


@dataclass
class DataConfig:
    """数据处理相关配置"""
    model_target_dir: str = "data/models_reproduce/residual-mse"
    dataset_path: str = "data/eopc04_14_IAU2000.62-now.csv"
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    val_mix: bool = False
    residual_type: str = "both"  # 'none', 'x', 'y', 'both'

    use_scaler: bool = True
    scaler_type: str = "minmax"  # 'minmax', 'standard', 'robust', 'none'
    scaler_after_residual: bool = False
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
    model_target_dir: str = "data/models_reproduce/residual-mse"
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    shuffle: bool = True
    early_stop: int = 5
    loss: str = "mae-corr"
    corr_alpha: float = 5e-4
    metrics: str = "loss"


def get_scaler_class(scaler_type: str):
    """根据类型字符串返回scaler类"""
    scaler_map = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "robust": RobustScaler,
        "none": None,
    }

    if scaler_type not in scaler_map:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    return scaler_map[scaler_type]


def load_config(json_path: str):
    """从 JSON 文件加载配置"""
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    model_cfg = ModelConfig(**cfg.get("model", {}))
    training_cfg = TrainingConfig(**cfg.get("training", {}))

    return data_cfg, model_cfg, training_cfg


# 默认从 config.json 加载
try:
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG = load_config("config.json")
except FileNotFoundError:
    print("⚠️ 未找到 config.json，使用默认配置")
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG = DataConfig(), ModelConfig(), TrainingConfig()
