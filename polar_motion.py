import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

from config import DATA_CONFIG


def get_polar_motion(path: str) -> List[Tuple[float, float]]:
    """
    读取极移数据并返回极移值列表
    
    参数:
        path: CSV文件路径
    
    返回:
        List[Tuple[float, float]]: 每个元组包含 (x_pole, y_pole)
    """
    # 读取极移数据
    data = pd.read_csv(path)
    
    # 提取极移值并组合成元组列表
    x_pole = data['x_pole'].values
    y_pole = data['y_pole'].values
    
    # 将两列数据组合成元组列表
    polar_motion_list = list(zip(x_pole, y_pole))
    
    return polar_motion_list


def compute_polar_motion_impact(data: pd.DataFrame) -> List[float]:
    """
    计算极移影响
    """

def dms_to_rad(dms_str):
    """
    将"度.分.秒"格式的字符串转换为弧度
    
    参数:
        dms_str: 字符串格式 "度.分.秒"
    
    返回:
        float: 弧度值
    """
    parts = dms_str.split('.')
    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0
    seconds = float(parts[2]) if len(parts) > 2 else 0
    
    # 转换为十进制度
    decimal_degrees = degrees + minutes/60 + seconds/3600
    
    # 转换为弧度
    radians = math.radians(decimal_degrees)
    
    return radians

def arcsec_to_rad(arcsec):
    """
    将角秒转换为弧度
    
    参数:
        arcsec: 角秒值
    
    返回:
        float: 弧度值
    """
    return arcsec * (math.pi / 180) / 3600

def calculate_polar_gravity_deviation(polar_motion, position):
    """
    计算极移引起的重力偏差
    
    参数:
        polar_motion: tuple of (x, y), 极移坐标，单位为角秒(as)
        position: tuple of (纬度, 经度), 测量点的坐标，格式为"度.分.秒"
        t: int or float, 时间索引（如果polar_motion是时间序列）
    
    返回:
        float: 重力偏差 δg_Polar(t)，单位为 uGal
    """
    # 常数定义
    R = 6371000  # 地球半径，单位：米
    OMEGA_E = 7.2921159e-5  # 地球自转角速度，单位：rad/s
    
    # 转换极移坐标从角秒到弧度
    x_as, y_as = polar_motion
    x_rad = arcsec_to_rad(x_as)
    y_rad = arcsec_to_rad(y_as)
    
    # 转换纬度和经度从"度.分.秒"格式到弧度
    phi = dms_to_rad(position[0])  # 纬度 φ
    lambda_ = dms_to_rad(position[1])  # 经度 λ
    
    # 计算重力偏差
    # δg_Polar(t) = 1.16 * R * Ω_E^2 * sin(2φ) * (x(t)cos(λ) - y(t)sin(λ)) * 10^8
    delta_g_polar = (1.16 * R * OMEGA_E**2 * math.sin(2*phi) * 
                     (x_rad * math.cos(lambda_) - y_rad * math.sin(lambda_)) * 1e8)
    
    return delta_g_polar


def plot_pm(pm_impact: List[float]) -> None:
    # 可视化，plt出pm_impact的曲线，使用黑色
    # 横坐标为时间，每个点为一日，从1962年1月1日开始计
    # 创建日期序列
    start_date = datetime(1962, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(len(pm_impact))]
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    plt.plot(dates, pm_impact, 'k-', linewidth=0.8, label='gPolar')
    
    # 设置标签和标题
    plt.xlabel('date', fontsize=12)
    plt.ylabel('gPolar (μGal)', fontsize=12)
    # plt.title('极移引起的重力偏差时间序列 (纬度18°, 经度109°)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    
    # 自动调整日期显示
    plt.gcf().autofmt_xdate()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形到文件
    output_path = Path(__file__).parent / 'figures' / 'polar_motion_impact.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存至: {output_path}")
    
    # 显示图形
    plt.show(block=True)
    
    # 打印统计信息
    print(f"数据点数量: {len(pm_impact)}")
    print(f"时间范围: {dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}")
    print(f"重力偏差范围: {min(pm_impact):.4f} ~ {max(pm_impact):.4f} μGal")
    print(f"平均重力偏差: {sum(pm_impact)/len(pm_impact):.4f} μGal")


if __name__ == "__main__":
    pm = get_polar_motion(DATA_CONFIG.dataset_path)
    position = ("18.15.0", "109.30.0")
    # position = ("40.0.0", "116.0.0")
    # pm_impact = [calculate_polar_gravity_deviation(i, position) for i in pm]
    # plot_pm(pm_impact)

    print(calculate_polar_gravity_deviation((0.108055,0.435614), position))
    print(calculate_polar_gravity_deviation((0.108055+0.06,0.435614+0.06), position))
    print(calculate_polar_gravity_deviation((0.108055+0.06,0.435614-0.06), position))
    print(calculate_polar_gravity_deviation((0.108055-0.06,0.435614+0.06), position))
    print(calculate_polar_gravity_deviation((0.108055-0.06,0.435614-0.06), position))

    
