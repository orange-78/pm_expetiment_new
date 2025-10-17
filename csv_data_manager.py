"""
CSV数据管理器 - csv_data_manager.py
用于读取和写入极移数据CSV文件（兼容同时存在 MJD 与 Year/Month/Day）
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta


class CSVDataManager:
    """CSV数据管理器 - 专门处理极移数据的读写"""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()

    # ========================== 基础功能 ==========================

    def _load_csv(self):
        """加载CSV文件并排序"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = ['x_pole', 'y_pole']
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列: {missing_cols}")

        # 同时存在 MJD 和日期列时确保一致性
        if {'Year', 'Month', 'Day'}.issubset(self.df.columns):
            self.df['_temp_date'] = pd.to_datetime(self.df[['Year', 'Month', 'Day']], errors='coerce')
            if 'MJD' not in self.df.columns:
                # 若缺MJD，自动补充
                self.df['MJD'] = (self.df['_temp_date'] - datetime(1858, 11, 17)).dt.days.astype(float)
        elif 'MJD' in self.df.columns:
            # 若只有MJD，生成日期
            base = datetime(1858, 11, 17)
            date_series = self.df['MJD'].apply(lambda mjd: base + timedelta(days=float(mjd)))
            self.df['Year'] = date_series.dt.year
            self.df['Month'] = date_series.dt.month
            self.df['Day'] = date_series.dt.day
            self.df['_temp_date'] = date_series
        else:
            raise ValueError("CSV必须包含 MJD 或 Year/Month/Day 至少一种时间信息。")

        # 排序逻辑：优先按 MJD 排序
        if self.df['MJD'].is_monotonic_increasing:
            self.df = self.df.sort_values('MJD')
        else:
            print("⚠️ 检测到 MJD 非单调递增，改用日期排序")
            self.df = self.df.sort_values('_temp_date')

        self.df = self.df.drop(columns=['_temp_date']).reset_index(drop=True)
        print(f"✅ 加载CSV: {self.csv_path.name}, 共 {len(self.df)} 行")

    def reload(self):
        self._load_csv()

    def get_total_length(self) -> int:
        return len(self.df)

    # ========================== 时间范围 ==========================

    def get_date_range(self) -> Tuple[str, str]:
        """获取数据的时间范围（同时返回MJD和日期格式）"""
        first_mjd, last_mjd = float(self.df['MJD'].iloc[0]), float(self.df['MJD'].iloc[-1])
        first_date = f"{int(self.df['Year'].iloc[0])}-{int(self.df['Month'].iloc[0]):02d}-{int(self.df['Day'].iloc[0]):02d}"
        last_date = f"{int(self.df['Year'].iloc[-1])}-{int(self.df['Month'].iloc[-1]):02d}-{int(self.df['Day'].iloc[-1]):02d}"
        return (f"MJD {first_mjd:.1f} ~ {last_mjd:.1f}", f"Date {first_date} ~ {last_date}")

    # ========================== 数据读取 ==========================

    def read_sequence_by_index(self, start_idx: int, length: int, return_dates: bool = False):
        end_idx = start_idx + length
        if start_idx < 0 or end_idx > len(self.df):
            raise ValueError(f"索引超出范围: start_idx={start_idx}, length={length}")

        subset = self.df.iloc[start_idx:end_idx]
        sequence = subset[['x_pole', 'y_pole']].values.astype(np.float32)

        if return_dates:
            return sequence, subset[['MJD', 'Year', 'Month', 'Day']].reset_index(drop=True)
        return sequence

    def read_sequence_by_mjd(self, start_mjd: float, length: int, return_dates: bool = False):
        idx = (self.df['MJD'] - start_mjd).abs().idxmin()
        return self.read_sequence_by_index(idx, length, return_dates)

    def read_sequence_by_date(self, start_date: Union[str, datetime], length: int, return_dates: bool = False):
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        mask = (
            (self.df['Year'] == start_date.year) &
            (self.df['Month'] == start_date.month) &
            (self.df['Day'] == start_date.day)
        )
        indices = self.df.index[mask].tolist()
        if not indices:
            raise ValueError(f"未找到日期: {start_date.strftime('%Y-%m-%d')}")
        start_idx = indices[0]
        return self.read_sequence_by_index(start_idx, length, return_dates)
    
    def read_latest_sequence(self, length: int, return_dates: bool = False):
        """
        获取最新的 length 条有效数据 (x_pole, y_pole)
        
        Args:
            length: 需要返回的序列长度
            return_dates: 是否同时返回时间信息 (MJD + Year/Month/Day)
        
        Returns:
            np.ndarray 或 (np.ndarray, pd.DataFrame)
        """
        # 找到最后一个有效索引
        valid_mask = self.df['x_pole'].notna() & self.df['y_pole'].notna()
        valid_indices = self.df.index[valid_mask]

        if len(valid_indices) == 0:
            raise ValueError("数据中没有有效的 x_pole / y_pole 数据")

        last_valid_idx = valid_indices[-1]
        start_idx = max(0, last_valid_idx - length + 1)

        subset = self.df.iloc[start_idx:last_valid_idx + 1]
        sequence = subset[['x_pole', 'y_pole']].values.astype(np.float32)

        if return_dates:
            dates_df = subset[['MJD', 'Year', 'Month', 'Day']].reset_index(drop=True)
            return sequence, dates_df

        return sequence


    # ========================== 写入预测 ==========================

    def _get_last_date(self) -> datetime:
        last_row = self.df.iloc[-1]
        return datetime(int(last_row['Year']), int(last_row['Month']), int(last_row['Day']))

    def _ensure_date_until(self, target_date: datetime):
        """自动扩展数据到目标日期，同时维护 MJD 与年月日"""
        last_date = self._get_last_date()
        if target_date <= last_date:
            return

        n_days = (target_date - last_date).days
        print(f"⚠️ 自动扩展 {n_days} 天日期到 {target_date.strftime('%Y-%m-%d')}")

        new_rows = []
        last_mjd = float(self.df['MJD'].iloc[-1])
        for i in range(1, n_days + 1):
            new_date = last_date + timedelta(days=i)
            new_rows.append({
                'MJD': last_mjd + i,
                'Year': new_date.year,
                'Month': new_date.month,
                'Day': new_date.day,
                'x_pole': np.nan,
                'y_pole': np.nan,
                'x_pole_predict': np.nan,
                'y_pole_predict': np.nan
            })

        self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

    def write_predictions(self,
                          predictions: np.ndarray,
                          start_date: Union[str, datetime, float, int],
                          date_format: str = "%Y-%m-%d",
                          overwrite: bool = False,
                          save_path: Optional[str] = None):
        """写入预测值，支持 MJD 或 日期起点"""
        if predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError("预测数据形状应为 (n_steps, 2)")

        n_steps = len(predictions)
        for col in ['x_pole_predict', 'y_pole_predict']:
            if col not in self.df.columns:
                self.df[col] = np.nan

        # 确定起始索引
        if isinstance(start_date, (float, int)):
            start_idx = (self.df['MJD'] - float(start_date)).abs().idxmin()
        else:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, date_format)
            self._ensure_date_until(start_date)
            mask = (
                (self.df['Year'] == start_date.year) &
                (self.df['Month'] == start_date.month) &
                (self.df['Day'] == start_date.day)
            )
            indices = self.df.index[mask].tolist()
            if not indices:
                raise ValueError(f"日期扩展后仍未找到起始日期: {start_date}")
            start_idx = indices[0]

        # 若末尾不足，扩展
        end_idx = start_idx + n_steps
        if end_idx > len(self.df):
            last_date = self._get_last_date()
            self._ensure_date_until(last_date + timedelta(days=end_idx - len(self.df)))

        # 写入预测
        write_indices = self.df.index[start_idx:end_idx]
        if not overwrite:
            mask = self.df.loc[write_indices, 'x_pole_predict'].isna()
            write_indices = write_indices[mask]

        actual_n_write = min(len(write_indices), n_steps)
        self.df.loc[write_indices[:actual_n_write], 'x_pole_predict'] = predictions[:actual_n_write, 0]
        self.df.loc[write_indices[:actual_n_write], 'y_pole_predict'] = predictions[:actual_n_write, 1]

        save_path = save_path or self.csv_path
        self.df.to_csv(save_path, index=False)
        print(f"✅ 写入 {actual_n_write} 行预测到 {save_path}")

    # ========================== 范围读取 ==========================

    def read_predictions_by_date_range(self,
                                       col0: str,
                                       col1: str,
                                       start_date: Union[str, datetime, float, int],
                                       end_date: Union[str, datetime, float, int],
                                       date_format: str = "%Y-%m-%d") -> np.ndarray:
        """读取指定日期或MJD范围的预测数据"""
        df = pd.read_csv(self.csv_path)

        # 优先使用 MJD
        if 'MJD' in df.columns and isinstance(start_date, (float, int)):
            mask = (df['MJD'] >= float(start_date)) & (df['MJD'] <= float(end_date))
            df_range = df.loc[mask, [col0, col1]].dropna()
        elif {'Year', 'Month', 'Day'}.issubset(df.columns):
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, date_format)
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, date_format)
            df['_temp_date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
            mask = (df['_temp_date'] >= start_date) & (df['_temp_date'] <= end_date)
            df_range = df.loc[mask, [col0, col1]].dropna()
            df = df.drop(columns=['_temp_date'])
        else:
            raise ValueError("CSV 文件中缺少 MJD 或 Year/Month/Day 列。")

        if df_range.empty:
            raise ValueError(f"在范围 {start_date} ~ {end_date} 内未找到有效数据。")

        return df_range.to_numpy(dtype=float)

    # ========================== 其他辅助 ==========================

    def _get_last_valid_index(self) -> int:
        mask = self.df['x_pole'].notna() & self.df['y_pole'].notna()
        if not mask.any():
            raise ValueError("无有效 x_pole / y_pole 数据")
        return mask[mask].index[-1]

    def append_predictions_from_last(self, predictions: np.ndarray, save_path: Optional[str] = None):
        last_valid_idx = self._get_last_valid_index()
        last_row = self.df.iloc[last_valid_idx]
        last_date = datetime(int(last_row['Year']), int(last_row['Month']), int(last_row['Day']))
        next_date = last_date + timedelta(days=1)
        print(f"🧩 从 {last_date.strftime('%Y-%m-%d')} 后开始追加预测")
        self.write_predictions(predictions, start_date=next_date, overwrite=True, save_path=save_path)

    def print_summary(self):
        mjd_range, date_range = self.get_date_range()
        print("\n" + "=" * 60)
        print("CSV数据摘要")
        print("=" * 60)
        print(f"文件路径: {self.csv_path}")
        print(f"共 {len(self.df)} 行")
        print(f"时间范围: {mjd_range} | {date_range}")
        print(f"列: {', '.join(self.df.columns)}")
        print("=" * 60 + "\n")

    def __repr__(self):
        return f"CSVDataManager('{self.csv_path.name}', {len(self.df)} rows)"
