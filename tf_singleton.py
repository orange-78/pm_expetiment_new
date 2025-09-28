"""
get tensorflow as tf
"""

import os
import random
import numpy as np

# 固定随机种子，确保可复现
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# 必须在导入 TensorFlow 之前设置可复现参数
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 如果你想禁用 oneDNN 浮点差异

import tensorflow as tf

tf.random.set_seed(SEED)
tf.experimental.numpy.experimental_enable_numpy_behavior()

# 可在这里做全局配置，比如禁用 GPU、设置显存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("[INFO] TensorFlow 初始化完成。版本:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 对外暴露 tf
__all__ = ["tf"]
