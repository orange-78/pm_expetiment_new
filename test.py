# import sys

# # 检查 TensorFlow 版本
# try:
#     import tensorflow as tf
#     print("TensorFlow version:", tf.__version__)
# except ImportError:
#     print("TensorFlow not installed")

# # 检查 keras 独立包版本
# try:
#     import keras
#     print("Standalone keras version:", keras.__version__)
# except ImportError:
#     print("Standalone keras not installed")

# # 检查 tf.keras 版本（它是 TensorFlow 自带的 keras）
# try:
#     print("tf.keras version:", tf.keras.__version__)
# except Exception as e:
#     print("tf.keras not available:", e)

# # 输出 Python 版本
# print("Python version:", sys.version)

import keras
import sys
import os

print("==== Keras 环境检测 ====")

# keras 包版本
print("keras.__version__:", getattr(keras, "__version__", "N/A"))

# keras 模块文件路径
print("keras.__file__:", keras.__file__)

# keras 模块所在的顶层包
print("keras package path:", os.path.dirname(keras.__file__))

# Python 环境信息
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# 尝试打印 Functional 类的路径
try:
    from keras.engine.functional import Functional
    print("Functional class (old API) loaded from:", Functional)
except Exception as e1:
    print("旧API Functional 不存在:", e1)

try:
    from keras.src.engine.functional import Functional
    print("Functional class (new API) loaded from:", Functional)
except Exception as e2:
    print("新API Functional 不存在:", e2)
