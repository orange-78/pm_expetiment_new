import sys

# 检查 TensorFlow 版本
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow not installed")

# 检查 keras 独立包版本
try:
    import keras
    print("Standalone keras version:", keras.__version__)
except ImportError:
    print("Standalone keras not installed")

# 检查 tf.keras 版本（它是 TensorFlow 自带的 keras）
try:
    print("tf.keras version:", tf.keras.__version__)
except Exception as e:
    print("tf.keras not available:", e)

# 输出 Python 版本
print("Python version:", sys.version)
