import json
import numpy as np
import pandas as pd
from pathlib import Path
import glob

from error_visualization import calculate_mae_by_step, calculate_mae_of_dataset

class PoleDataProcessor:
    """处理极移预测数据和真实数据的类"""
    
    def __init__(self, prediction_folder, history_file):
        """
        初始化处理器
        
        参数:
            prediction_folder: str, 包含预测CSV文件的文件夹路径
            history_file: str, 历史真实数据CSV文件路径
        """
        self.prediction_folder = prediction_folder
        self.history_file = history_file
        self.pred_data = None
        self.true_data = None
        
    def load_prediction_data(self):
        """
        加载并处理所有预测数据
        
        返回:
            pred_data: ndarray, 形状为 (m, n, 2)，其中n为最短文件的数据长度
        """
        # 获取文件夹中所有CSV文件
        csv_files = glob.glob(str(Path(self.prediction_folder) / "*.csv"))
        
        if not csv_files:
            raise ValueError(f"在文件夹 {self.prediction_folder} 中未找到CSV文件")
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 存储每个文件的数据和起始MJD
        file_data_list = []
        
        for csv_file in csv_files:
            try:
                # 读取CSV文件，跳过空行
                df = pd.read_csv(csv_file, skip_blank_lines=True)
                
                # 删除完全为空的行
                df = df.dropna(how='all')
                
                # 如果dataframe为空，跳过
                if len(df) == 0:
                    print(f"警告: {Path(csv_file).name} 读取后无有效数据，跳过此文件")
                    continue
                
                # 检查必需的列是否存在
                required_cols = ['MJD', 'Type', 'x_pole', 'y_pole']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"警告: {Path(csv_file).name} 缺少列 {missing_cols}，跳过此文件")
                    continue
                
                # 筛选Type为prediction的数据
                pred_df = df[df['Type'] == 'prediction'].copy()
                
                if len(pred_df) == 0:
                    print(f"警告: {Path(csv_file).name} 中没有Type=prediction的数据，跳过此文件")
                    continue
                
                # 删除关键列中有空值的行
                pred_df = pred_df.dropna(subset=['MJD', 'x_pole', 'y_pole'])
                
                if len(pred_df) == 0:
                    print(f"警告: {Path(csv_file).name} 中过滤空值后无有效数据，跳过此文件")
                    continue
                
                # 按MJD排序确保数据有序
                pred_df = pred_df.sort_values('MJD').reset_index(drop=True)
                
                # 提取x_pole和y_pole数据
                xy_data = pred_df[['x_pole', 'y_pole']].values
                mjd_data = pred_df['MJD'].values
                start_mjd = mjd_data[0]
                
                file_data_list.append({
                    'filename': Path(csv_file).name,
                    'start_mjd': start_mjd,
                    'mjd': mjd_data,
                    'data': xy_data,
                    'length': len(xy_data)
                })
                
                print(f"  {Path(csv_file).name}: {len(xy_data)} 条数据, 起始MJD={start_mjd}")
                
            except Exception as e:
                print(f"错误: 读取文件 {csv_file} 时出错: {str(e)}")
                continue
        
        if not file_data_list:
            raise ValueError("没有成功读取任何有效的预测数据文件")
        
        # 按起始MJD排序
        file_data_list.sort(key=lambda x: x['start_mjd'])
        
        # 找到最短的数据长度
        min_length = min(item['length'] for item in file_data_list)
        print(f"\n最短数据长度: {min_length}")
        
        # 构建最终的pred_data数组 (m, min_length, 2)
        m = len(file_data_list)
        pred_data = np.zeros((m, min_length, 2))
        self.mjd_arrays = []  # 保存每个文件对应的MJD值用于匹配真实数据
        
        for i, item in enumerate(file_data_list):
            pred_data[i] = item['data'][:min_length]
            self.mjd_arrays.append(item['mjd'][:min_length])
            print(f"  文件 {i+1}: {item['filename']}, MJD范围 [{item['start_mjd']}, {item['mjd'][min_length-1]}]")
        
        self.pred_data = pred_data
        print(f"\n预测数据形状: {pred_data.shape}")
        return pred_data
    
    def load_true_data(self):
        """
        根据预测数据的MJD从历史数据中加载对应的真实数据
        
        返回:
            true_data: ndarray, 形状与pred_data一致
        """
        if self.pred_data is None:
            raise ValueError("请先调用 load_prediction_data() 加载预测数据")
        
        # 读取历史真实数据，跳过空行
        history_df = pd.read_csv(self.history_file, skip_blank_lines=True)
        
        # 删除完全为空的行
        history_df = history_df.dropna(how='all')
        
        if len(history_df) == 0:
            raise ValueError("历史数据文件读取后无有效数据")
        
        # 检查必需的列
        required_cols = ['MJD', 'x_pole', 'y_pole']
        missing_cols = [col for col in required_cols if col not in history_df.columns]
        if missing_cols:
            raise ValueError(f"历史数据文件缺少列: {missing_cols}")
        
        # 删除关键列中有空值的行
        history_df = history_df.dropna(subset=required_cols)
        
        print(f"\n历史数据总计: {len(history_df)} 条有效记录")
        
        # 创建MJD到数据的映射字典，提高查询效率
        history_dict = {}
        for _, row in history_df.iterrows():
            mjd = row['MJD']
            history_dict[mjd] = [row['x_pole'], row['y_pole']]
        
        # 构建true_data数组
        m, n, _ = self.pred_data.shape
        true_data = np.full((m, n, 2), np.nan)  # 用NaN初始化，未匹配的数据保持为NaN
        
        matched_count = 0
        total_count = m * n
        
        for i in range(m):
            for j in range(n):
                mjd = self.mjd_arrays[i][j]
                if mjd in history_dict:
                    true_data[i, j] = history_dict[mjd]
                    matched_count += 1
        
        print(f"匹配成功: {matched_count}/{total_count} ({100*matched_count/total_count:.1f}%)")
        
        if matched_count == 0:
            print("警告: 没有匹配到任何真实数据，请检查MJD值是否对应")
        
        self.true_data = true_data
        print(f"真实数据形状: {true_data.shape}")
        return true_data
    
    def save_arrays(self, pred_output='pred_data.npy', true_output='true_data.npy'):
        """
        保存处理后的数组到文件
        
        参数:
            pred_output: str, 预测数据保存路径
            true_output: str, 真实数据保存路径
        """
        if self.pred_data is None or self.true_data is None:
            raise ValueError("请先加载预测数据和真实数据")
        
        np.save(pred_output, self.pred_data)
        np.save(true_output, self.true_data)
        print(f"\n数据已保存:")
        print(f"  预测数据: {pred_output} (形状: {self.pred_data.shape})")
        print(f"  真实数据: {true_output} (形状: {self.true_data.shape})")
    
    def load_arrays(self, pred_input='pred_data.npy', true_input='true_data.npy'):
        """
        从文件加载保存的数组
        
        参数:
            pred_input: str, 预测数据文件路径
            true_input: str, 真实数据文件路径
        
        返回:
            pred_data, true_data: 加载的数组
        """
        self.pred_data = np.load(pred_input)
        self.true_data = np.load(true_input)
        print(f"数据已加载:")
        print(f"  预测数据形状: {self.pred_data.shape}")
        print(f"  真实数据形状: {self.true_data.shape}")
        return self.pred_data, self.true_data


# 使用示例
if __name__ == "__main__":
    # 设置路径
    prediction_folder = "data/models_baseline/IERS/BulletinA16-25"  # 预测数据文件夹
    history_file = "data/eopc04_14_IAU2000.62-now.csv"     # 历史真实数据文件
    
    # 创建处理器实例
    processor = PoleDataProcessor(prediction_folder, history_file)
    
    # 加载并处理预测数据
    pred_data = processor.load_prediction_data()
    
    # 加载对应的真实数据
    true_data = processor.load_true_data()
    
    # 保存数组
    processor.save_arrays('data/models_baseline/IERS/BulletinA16-25/pred_data.npy', 'data/models_baseline/IERS/BulletinA16-25/true_data.npy')
    
    # 后续可以这样加载保存的数组
    # processor.load_arrays('pred_data.npy', 'true_data.npy')

    # 绘制mae by day
    mae_by_days = calculate_mae_by_step(processor.true_data, processor.pred_data)
    total_mae_by_day = calculate_mae_of_dataset(mae_by_days)
    mae_list = total_mae_by_day.tolist()  # 转换为Python列表
    json_str = json.dumps(mae_list)  # 转换为JSON字符串

    # 保存到文件
    with open('data/models_baseline/IERS/BulletinA16-25/mae_result.json', 'w') as f:
        json.dump(mae_list, f, indent=2)
    
    # 计算一些基本统计信息
    print("\n=== 数据统计 ===")
    print(f"预测数据均值: x_pole={np.mean(pred_data[:,:,0]):.6f}, y_pole={np.mean(pred_data[:,:,1]):.6f}")
    print(f"真实数据均值: x_pole={np.nanmean(true_data[:,:,0]):.6f}, y_pole={np.nanmean(true_data[:,:,1]):.6f}")
    
    # 计算预测误差（忽略NaN值）
    diff = pred_data - true_data
    mae_x = np.nanmean(np.abs(diff[:,:,0]))
    mae_y = np.nanmean(np.abs(diff[:,:,1]))
    print(f"平均绝对误差: x_pole={mae_x:.6f}, y_pole={mae_y:.6f}")