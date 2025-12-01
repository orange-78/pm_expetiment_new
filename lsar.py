
from data_pipeline import DataLoader, DataSplitter


_, raw_history_data, test_data = DataSplitter.time_split(DataLoader.load_xy_from_csv, 0.0, 0.85)

history_data = raw_history_data[-3650:]

# history_data 为shape (N, 2) 的 numpy 数组 (x_pole, y_pole)用于数据拟合
# test_data 为 shape (M, 2) 的 numpy 数组 (x_pole, y_pole)用于测试模型性能

# 构建一个LS+AR模型，使用历史数据进行训练，并在测试数据上评估模型性能

# 要将代码分为模型拟合、模型预测和模型评估三部分，预测结果和评估结果支持中间状态保存。保存格式待定

