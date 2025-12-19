import numpy as np
from scipy.optimize import least_squares
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
import pickle
import json
from typing import Tuple, Dict, Optional
from model_tester import ModelEvaluator
from error_visualization import calculate_mae_by_step, calculate_mae_of_dataset, plot_mae_by_step
from visualizer import plot_pm

class LSARModel:
    """
    LS+AR极移预测模型
    LS: 最小二乘拟合 (趋势 + 钱德勒摆动 + 年周期)
    AR: 自回归模型 (对残差建模)
    """
    
    def __init__(self, ar_order: Optional[int] = None, fit_periods: bool = False):
        """
        初始化模型
        
        Parameters:
        -----------
        ar_order : int, optional
            AR模型阶数，如果为None则自动选择
        fit_periods : bool, default=False
            是否将周期也作为拟合参数
            False: 使用固定周期 (P_C=365.25天, P_A=433天)
            True: 将周期作为拟合参数
        """
        self.ar_order = ar_order
        self.fit_periods = fit_periods
        
        # LS拟合参数
        self.ls_params_x = None  # x方向的LS参数
        self.ls_params_y = None  # y方向的LS参数
        
        # AR模型
        self.ar_model_x = None
        self.ar_model_y = None
        
        # 训练数据信息
        self.t_start = 0
        self.t_end = 0
        self.residuals_x = None
        self.residuals_y = None
        
        # 拟合质量指标
        self.fit_metrics = {}
        
    def _ls_model(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        LS模型函数
        
        固定周期模式 (fit_periods=False):
        PM = a0 + a1*t + a2*cos(2πt/P_C + φ_C) + a3*cos(2πt/P_A + φ_A)
        params: [a0, a1, a2, φ_C, a3, φ_A]
        
        拟合周期模式 (fit_periods=True):
        PM = a0 + a1*t + a2*cos(2πt/P_C + φ_C) + a3*cos(2πt/P_A + φ_A)
        params: [a0, a1, a2, φ_C, P_C, a3, φ_A, P_A]
        
        Parameters:
        -----------
        t : np.ndarray
            时间序列
        params : np.ndarray
            参数数组
            
        Returns:
        --------
        np.ndarray
            模型预测值
        """
        if self.fit_periods:
            # 拟合周期模式：8个参数
            a0, a1, a2, phi_c, P_C, a3, phi_a, P_A = params
        else:
            # 固定周期模式：6个参数
            a0, a1, a2, phi_c, a3, phi_a = params
            P_C = 365.25  # 年周期 (天)
            P_A = 433.0   # 钱德勒摆动周期 (天)
        
        trend = a0 + a1 * t
        annual = a2 * np.cos(2 * np.pi * t / P_C + phi_c)
        chandler = a3 * np.cos(2 * np.pi * t / P_A + phi_a)
        
        return trend + annual + chandler
    
    def _ls_residuals(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算LS拟合残差"""
        return y - self._ls_model(t, params)
    
    def _fit_ls(self, t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个分量进行LS拟合
        
        Parameters:
        -----------
        t : np.ndarray
            时间序列
        y : np.ndarray
            极移数据
            
        Returns:
        --------
        params : np.ndarray
            拟合参数
        residuals : np.ndarray
            残差序列
        """
        # 初始参数估计
        a0_init = np.mean(y)
        a1_init = (y[-1] - y[0]) / (t[-1] - t[0])
        a2_init = np.std(y) * 0.5  # 年周期振幅
        phi_c_init = 0.0
        a3_init = np.std(y) * 0.5  # 钱德勒摆动振幅
        phi_a_init = 0.0
        
        if self.fit_periods:
            # 拟合周期模式：8个参数
            P_C_init = 365.25  # 年周期初值
            P_A_init = 433.0   # 钱德勒摆动周期初值
            initial_params = np.array([
                a0_init, a1_init, a2_init, phi_c_init, P_C_init, 
                a3_init, phi_a_init, P_A_init
            ])
            
            # 设置参数边界，防止周期偏离太远
            bounds_lower = [-np.inf, -np.inf, 0, -2*np.pi, 360, 0, -2*np.pi, 420]
            bounds_upper = [np.inf, np.inf, np.inf, 2*np.pi, 370, np.inf, 2*np.pi, 445]
            bounds = (bounds_lower, bounds_upper)
        else:
            # 固定周期模式：6个参数
            initial_params = np.array([a0_init, a1_init, a2_init, phi_c_init, a3_init, phi_a_init])
            bounds = (-np.inf, np.inf)
        
        # 非线性最小二乘拟合
        result = least_squares(
            self._ls_residuals,
            initial_params,
            args=(t, y),
            bounds=bounds,
            method='trf' if self.fit_periods else 'lm',
            max_nfev=10000
        )
        
        params = result.x
        residuals = self._ls_residuals(params, t, y)
        
        return params, residuals
    
    def _select_ar_order(self, residuals: np.ndarray, max_order: int = 30) -> int:
        """
        使用AIC准则自动选择AR阶数
        
        Parameters:
        -----------
        residuals : np.ndarray
            残差序列
        max_order : int
            最大考虑的阶数
            
        Returns:
        --------
        int
            最优阶数
        """
        aic_values = []
        
        for order in range(1, min(max_order + 1, len(residuals) // 10)):
            try:
                model = AutoReg(residuals, lags=order, old_names=False)
                model_fit = model.fit()
                aic_values.append(model_fit.aic)
            except:
                aic_values.append(np.inf)
        
        optimal_order = np.argmin(aic_values) + 1
        return optimal_order
    
    def fit(self, data: np.ndarray, verbose: bool = True) -> Dict:
        """
        拟合LS+AR模型
        
        Parameters:
        -----------
        data : np.ndarray, shape (N, 2)
            历史极移数据 [x_pole, y_pole]
        verbose : bool
            是否打印拟合信息
            
        Returns:
        --------
        dict
            拟合结果和指标
        """
        N = len(data)
        t = np.arange(N)
        
        self.t_start = 0
        self.t_end = N - 1
        
        x_pole = data[:, 0]
        y_pole = data[:, 1]
        
        # 1. LS拟合 - X方向
        if verbose:
            print("正在拟合X方向的LS模型...")
        self.ls_params_x, self.residuals_x = self._fit_ls(t, x_pole)
        
        # 2. LS拟合 - Y方向
        if verbose:
            print("正在拟合Y方向的LS模型...")
        self.ls_params_y, self.residuals_y = self._fit_ls(t, y_pole)
        
        # 3. AR模型拟合 - X方向
        if self.ar_order is None:
            ar_order_x = self._select_ar_order(self.residuals_x)
            if verbose:
                print(f"X方向自动选择AR阶数: {ar_order_x}")
        else:
            ar_order_x = self.ar_order
        
        self.ar_model_x = AutoReg(self.residuals_x, lags=ar_order_x, old_names=False)
        self.ar_fit_x = self.ar_model_x.fit()
        
        # 4. AR模型拟合 - Y方向
        if self.ar_order is None:
            ar_order_y = self._select_ar_order(self.residuals_y)
            if verbose:
                print(f"Y方向自动选择AR阶数: {ar_order_y}")
        else:
            ar_order_y = self.ar_order
        
        self.ar_model_y = AutoReg(self.residuals_y, lags=ar_order_y, old_names=False)
        self.ar_fit_y = self.ar_model_y.fit()
        
        # 5. 计算拟合指标
        # AR模型的fittedvalues会因为滞后阶数而短于原始序列
        # 需要对齐LS预测值和AR拟合值
        ls_pred_x = self._ls_model(t, self.ls_params_x)
        ls_pred_y = self._ls_model(t, self.ls_params_y)
        
        # AR模型从第 ar_order 个点开始有fittedvalues
        # 构建完整的拟合序列
        x_fitted = np.full(N, np.nan)
        y_fitted = np.full(N, np.nan)
        
        # LS部分在整个时间段都有值
        x_fitted[:] = ls_pred_x
        y_fitted[:] = ls_pred_y
        
        # AR部分只在有fittedvalues的位置加上
        ar_start_x = ar_order_x
        ar_start_y = ar_order_y
        x_fitted[ar_start_x:] += self.ar_fit_x.fittedvalues
        y_fitted[ar_start_y:] += self.ar_fit_y.fittedvalues
        
        # 只在两个方向都有完整拟合的区域计算指标
        start_idx = max(ar_order_x, ar_order_y)
        x_pole_trim = x_pole[start_idx:]
        y_pole_trim = y_pole[start_idx:]
        x_fitted_trim = x_fitted[start_idx:]
        y_fitted_trim = y_fitted[start_idx:]
        
        # 计算R²和RMSE
        ss_res_x = np.sum((x_pole_trim - x_fitted_trim) ** 2)
        ss_tot_x = np.sum((x_pole_trim - np.mean(x_pole_trim)) ** 2)
        r2_x = 1 - ss_res_x / ss_tot_x
        rmse_x = np.sqrt(np.mean((x_pole_trim - x_fitted_trim) ** 2))
        
        ss_res_y = np.sum((y_pole_trim - y_fitted_trim) ** 2)
        ss_tot_y = np.sum((y_pole_trim - np.mean(y_pole_trim)) ** 2)
        r2_y = 1 - ss_res_y / ss_tot_y
        rmse_y = np.sqrt(np.mean((y_pole_trim - y_fitted_trim) ** 2))
        
        self.fit_metrics = {
            'ar_order_x': int(ar_order_x),
            'ar_order_y': int(ar_order_y),
            'r2_x': float(r2_x),
            'r2_y': float(r2_y),
            'rmse_x': float(rmse_x),
            'rmse_y': float(rmse_y),
            'aic_x': float(self.ar_fit_x.aic),
            'aic_y': float(self.ar_fit_y.aic),
            'ls_params_x': self.ls_params_x.tolist(),
            'ls_params_y': self.ls_params_y.tolist(),
        }
        
        if verbose:
            print("\n===== 拟合结果 =====")
            print(f"X方向: R² = {r2_x:.6f}, RMSE = {rmse_x:.6f} mas")
            print(f"Y方向: R² = {r2_y:.6f}, RMSE = {rmse_y:.6f} mas")
            if self.fit_periods:
                print(f"\nX方向LS参数 (含周期):")
                print(f"  趋势: a0={self.ls_params_x[0]:.4f}, a1={self.ls_params_x[1]:.6f}")
                print(f"  年周期: a2={self.ls_params_x[2]:.4f}, φ_C={self.ls_params_x[3]:.4f}, P_C={self.ls_params_x[4]:.2f}天")
                print(f"  钱德勒: a3={self.ls_params_x[5]:.4f}, φ_A={self.ls_params_x[6]:.4f}, P_A={self.ls_params_x[7]:.2f}天")
                print(f"\nY方向LS参数 (含周期):")
                print(f"  趋势: a0={self.ls_params_y[0]:.4f}, a1={self.ls_params_y[1]:.6f}")
                print(f"  年周期: a2={self.ls_params_y[2]:.4f}, φ_C={self.ls_params_y[3]:.4f}, P_C={self.ls_params_y[4]:.2f}天")
                print(f"  钱德勒: a3={self.ls_params_y[5]:.4f}, φ_A={self.ls_params_y[6]:.4f}, P_A={self.ls_params_y[7]:.2f}天")
            else:
                print(f"X方向LS参数: {self.ls_params_x}")
                print(f"Y方向LS参数: {self.ls_params_y}")
        
        return self.fit_metrics
    
    def save_model(self, filepath: str):
        """保存模型到文件"""
        model_data = {
            'ar_order': self.ar_order,
            'fit_periods': self.fit_periods,
            'ls_params_x': self.ls_params_x,
            'ls_params_y': self.ls_params_y,
            't_start': self.t_start,
            't_end': self.t_end,
            'residuals_x': self.residuals_x,
            'residuals_y': self.residuals_y,
            'ar_fit_x': self.ar_fit_x,
            'ar_fit_y': self.ar_fit_y,
            'fit_metrics': self.fit_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ar_order = model_data['ar_order']
        self.fit_periods = model_data.get('fit_periods', False)  # 兼容旧版本
        self.ls_params_x = model_data['ls_params_x']
        self.ls_params_y = model_data['ls_params_y']
        self.t_start = model_data['t_start']
        self.t_end = model_data['t_end']
        self.residuals_x = model_data['residuals_x']
        self.residuals_y = model_data['residuals_y']
        self.ar_fit_x = model_data['ar_fit_x']
        self.ar_fit_y = model_data['ar_fit_y']
        self.fit_metrics = model_data['fit_metrics']
        
        print(f"模型已从 {filepath} 加载")
    
    def save_metrics(self, filepath: str):
        """保存拟合指标到JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(self.fit_metrics, f, indent=4)
        
        print(f"拟合指标已保存到: {filepath}")
    
    def predict(self, n_steps: int) -> np.ndarray:
        """
        多步预测
        
        Parameters:
        -----------
        n_steps : int
            预测步长（天数）
            
        Returns:
        --------
        np.ndarray, shape (n_steps, 2)
            预测结果 [x_pole, y_pole]
        """
        if self.ls_params_x is None or self.ar_fit_x is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        # 预测时间点（从训练数据结束后开始）
        t_pred = np.arange(self.t_end + 1, self.t_end + 1 + n_steps)
        
        # LS部分预测（确定性成分）
        ls_pred_x = self._ls_model(t_pred, self.ls_params_x)
        ls_pred_y = self._ls_model(t_pred, self.ls_params_y)
        
        # AR部分预测（随机性成分）
        # 使用AR模型的forecast方法进行多步预测
        ar_pred_x = self.ar_fit_x.forecast(steps=n_steps)
        ar_pred_y = self.ar_fit_y.forecast(steps=n_steps)
        
        # 组合预测结果
        x_pred = ls_pred_x + ar_pred_x
        y_pred = ls_pred_y + ar_pred_y
        
        # 返回 (n_steps, 2) 的数组
        predictions = np.column_stack([x_pred, y_pred])
        
        return predictions
    
    def predict_with_uncertainty(self, n_steps: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带置信区间的预测
        
        Parameters:
        -----------
        n_steps : int
            预测步长
        alpha : float, default=0.05
            显著性水平，alpha=0.05 对应 95% 置信区间
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_steps, 2)
            预测均值
        lower_bound : np.ndarray, shape (n_steps, 2)
            置信区间下界
        upper_bound : np.ndarray, shape (n_steps, 2)
            置信区间上界
        """
        if self.ls_params_x is None or self.ar_fit_x is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        # 预测时间点
        t_pred = np.arange(self.t_end + 1, self.t_end + 1 + n_steps)
        
        # LS部分预测（确定性成分，无不确定性）
        ls_pred_x = self._ls_model(t_pred, self.ls_params_x)
        ls_pred_y = self._ls_model(t_pred, self.ls_params_y)
        
        # AR部分预测（带置信区间）
        ar_forecast_x = self.ar_fit_x.forecast(steps=n_steps)
        ar_forecast_y = self.ar_fit_y.forecast(steps=n_steps)
        
        # 计算AR预测的标准误差
        # 多步预测的方差会随步长增加
        from scipy import stats
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # 获取AR模型的残差标准差
        sigma_x = np.sqrt(self.ar_fit_x.sigma2)
        sigma_y = np.sqrt(self.ar_fit_y.sigma2)
        
        # 多步预测标准误差（简化版本，假设误差累积）
        # 更精确的方法需要考虑AR系数的影响
        se_x = sigma_x * np.sqrt(np.arange(1, n_steps + 1))
        se_y = sigma_y * np.sqrt(np.arange(1, n_steps + 1))
        
        # 组合预测
        x_pred = ls_pred_x + ar_forecast_x
        y_pred = ls_pred_y + ar_forecast_y
        
        # 计算置信区间（只考虑AR部分的不确定性）
        x_lower = x_pred - z_score * se_x
        x_upper = x_pred + z_score * se_x
        y_lower = y_pred - z_score * se_y
        y_upper = y_pred + z_score * se_y
        
        predictions = np.column_stack([x_pred, y_pred])
        lower_bound = np.column_stack([x_lower, y_lower])
        upper_bound = np.column_stack([x_upper, y_upper])
        
        return predictions, lower_bound, upper_bound
    
    def save_predictions(self, predictions: np.ndarray, filepath: str):
        """
        保存预测结果
        
        Parameters:
        -----------
        predictions : np.ndarray, shape (n_steps, 2)
            预测结果
        filepath : str
            保存路径（.npy格式）
        """
        np.save(filepath, predictions)
        print(f"预测结果已保存到: {filepath}")
    
    def save_predictions_csv(self, predictions: np.ndarray, filepath: str):
        """
        保存预测结果为CSV格式
        
        Parameters:
        -----------
        predictions : np.ndarray, shape (n_steps, 2)
            预测结果
        filepath : str
            保存路径（.csv格式）
        """
        import pandas as pd
        df = pd.DataFrame(predictions, columns=['x_pole', 'y_pole'])
        df.index.name = 'step'
        df.to_csv(filepath)
        print(f"预测结果已保存到: {filepath}")


class LSARBatchTester:
    """
    LS+AR模型批量测试器（滑动窗口测试）
    """
    
    def __init__(self, model_class=LSARModel, ar_order=None, fit_periods=False):
        """
        初始化批量测试器
        
        Parameters:
        -----------
        model_class : class
            模型类
        ar_order : int, optional
            AR阶数
        fit_periods : bool
            是否拟合周期参数
        """
        self.model_class = model_class
        self.ar_order = ar_order
        self.fit_periods = fit_periods
    
    def prepare_sliding_window_data(self, 
                                    raw_history_data: np.ndarray,
                                    test_data: np.ndarray,
                                    m: int,
                                    n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备滑动窗口测试数据
        
        Parameters:
        -----------
        raw_history_data : np.ndarray, shape (N_hist, 2)
            原始历史数据
        test_data : np.ndarray, shape (N_test, 2)
            测试数据
        m : int
            每个模型使用的拟合数据长度
        n : int
            预测步长
            
        Returns:
        --------
        batch_history : np.ndarray, shape (batchsize, m, 2)
            批量历史数据（用于拟合）
        batch_truth : np.ndarray, shape (batchsize, n, 2)
            批量测试真实数据
        """
        N_test = len(test_data)
        batchsize = N_test - n + 1
        
        if batchsize <= 0:
            raise ValueError(f"测试数据长度({N_test})必须 >= 预测步长({n})")
        
        # 初始化批量数据
        batch_history = np.zeros((batchsize, m, 2))
        batch_truth = np.zeros((batchsize, n, 2))
        
        # 对于每个测试样本
        for i in range(batchsize):
            # 第i个测试数据的真实值：test_data[i:i+n]
            batch_truth[i] = test_data[i:i+n]
            
            # 第i个测试数据对应模型的拟合数据：
            # - 需要m条数据，这m条数据应该紧邻预测起点(test_data[i])之前
            # - 当 i < m: 部分来自raw_history_data，部分来自test_data
            # - 当 i >= m: 完全来自test_data[i-m:i]
            
            if i < m:
                # 需要从history补充数据
                from_history = m - i  # 从history取多少条
                from_test = i          # 从test取多少条
                
                # 前面的from_history条来自raw_history_data的最后from_history条
                batch_history[i, :from_history] = raw_history_data[-from_history:]
                
                # 后面的from_test条来自test_data的前from_test条
                if from_test > 0:
                    batch_history[i, from_history:] = test_data[:from_test]
            else:
                # i >= m: 完全从test_data中取紧邻预测点之前的m条
                batch_history[i] = test_data[i-m:i]
        
        return batch_history, batch_truth
    
    def batch_test(self,
                   raw_history_data: np.ndarray,
                   test_data: np.ndarray,
                   m: int,
                   n: int,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量测试（滑动窗口）
        
        Parameters:
        -----------
        raw_history_data : np.ndarray, shape (N_hist, 2)
            原始历史数据
        test_data : np.ndarray, shape (N_test, 2)
            测试数据
        m : int
            拟合数据长度
        n : int
            预测步长
        verbose : bool
            是否显示进度
            
        Returns:
        --------
        batch_history : np.ndarray, shape (batchsize, m, 2)
            批量历史数据
        batch_truth : np.ndarray, shape (batchsize, n, 2)
            批量测试真实数据
        batch_predictions : np.ndarray, shape (batchsize, n, 2)
            批量预测数据
        """
        # 准备数据
        batch_history, batch_truth = self.prepare_sliding_window_data(
            raw_history_data, test_data, m, n
        )
        
        batchsize = len(batch_history)
        batch_predictions = np.zeros((batchsize, n, 2))
        
        if verbose:
            print(f"\n开始批量测试:")
            print(f"  - 批次大小: {batchsize}")
            print(f"  - 拟合数据长度: {m}")
            print(f"  - 预测步长: {n}")
            print(f"  - AR阶数: {self.ar_order if self.ar_order else '自动选择'}")
            print(f"  - 拟合周期: {self.fit_periods}\n")
        
        # 对每个批次进行拟合和预测
        for i in range(batchsize):
            if verbose and (i + 1) % max(1, batchsize // 10) == 0:
                print(f"进度: {i+1}/{batchsize} ({100*(i+1)/batchsize:.1f}%)")
            
            # 创建并拟合模型
            model = self.model_class(
                ar_order=self.ar_order,
                fit_periods=self.fit_periods
            )
            model.fit(batch_history[i], verbose=False)
            
            # 预测
            batch_predictions[i] = model.predict(n)
        
        if verbose:
            print(f"批量测试完成!\n")
        
        return batch_history, batch_truth, batch_predictions
    
    def save_batch_results(self,
                          batch_history: np.ndarray,
                          batch_truth: np.ndarray,
                          batch_predictions: np.ndarray,
                          output_dir: str):
        """
        保存批量测试结果
        
        Parameters:
        -----------
        batch_history : np.ndarray
            批量历史数据
        batch_truth : np.ndarray
            批量真实数据
        batch_predictions : np.ndarray
            批量预测数据
        output_dir : str
            输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        np.save(os.path.join(output_dir, f'{str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_history.npy'), batch_history)
        np.save(os.path.join(output_dir, f'{str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_truth.npy'), batch_truth)
        np.save(os.path.join(output_dir, f'{str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_predictions.npy'), batch_predictions)
        
        # 保存元数据
        metadata = {
            'batchsize': int(batch_history.shape[0]),
            'm': int(batch_history.shape[1]),
            'n': int(batch_predictions.shape[1]),
            'ar_order': self.ar_order,
            'fit_periods': self.fit_periods,
            'shapes': {
                'batch_history': list(batch_history.shape),
                'batch_truth': list(batch_truth.shape),
                'batch_predictions': list(batch_predictions.shape)
            }
        }
        
        with open(os.path.join(output_dir, f'{str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"批量测试结果已保存到: {output_dir}")
        print(f"  - {str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_history.npy: {batch_history.shape}")
        print(f"  - {str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_truth.npy: {batch_truth.shape}")
        print(f"  - {str(int(batch_history.shape[1]))}_{str(int(batch_predictions.shape[1]))}batch_predictions.npy: {batch_predictions.shape}")


class LSARModelPredictor:
    """
    LS+AR模型预测器（独立的预测类）
    """
    
    def __init__(self, model: LSARModel):
        """
        初始化预测器
        
        Parameters:
        -----------
        model : LSARModel
            已拟合的模型
        """
        self.model = model
        self.prediction_cache = {}
    
    def batch_predict(self, step_lengths: list) -> Dict[int, np.ndarray]:
        """
        批量预测多个步长
        
        Parameters:
        -----------
        step_lengths : list of int
            预测步长列表，例如 [1, 7, 30, 90]
            
        Returns:
        --------
        dict
            {步长: 预测结果数组}
        """
        results = {}
        max_steps = max(step_lengths)
        
        # 一次性预测最大步长
        all_predictions = self.model.predict(max_steps)
        
        # 提取各个步长的结果
        for steps in step_lengths:
            results[steps] = all_predictions[:steps, :]
            self.prediction_cache[steps] = results[steps]
        
        return results
    
    def save_batch_predictions(self, predictions_dict: Dict[int, np.ndarray], 
                              output_dir: str):
        """
        保存批量预测结果
        
        Parameters:
        -----------
        predictions_dict : dict
            {步长: 预测结果}
        output_dir : str
            输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for steps, pred in predictions_dict.items():
            filepath = os.path.join(output_dir, f'predictions_{steps}days.npy')
            np.save(filepath, pred)
        
        # 保存元数据
        metadata = {
            'step_lengths': list(predictions_dict.keys()),
            'prediction_shape': {k: v.shape for k, v in predictions_dict.items()},
            'model_type': 'LS+AR',
            'fit_periods': self.model.fit_periods
        }
        
        metadata_path = os.path.join(output_dir, 'predictions_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"批量预测结果已保存到: {output_dir}")


def run_sliding_window_test(
    raw_history_data: np.ndarray,
    test_data: np.ndarray,
    output_dir: str,
    m: int = 2000,
    n: int = 400,
    ar_order: Optional[int] = None,
    fit_periods: bool = True,
    verbose: bool = True,
    tolerance: float = 0.063
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    执行滑动窗口批量测试
    
    参数:
        raw_history_data: 原始历史数据
        test_data: 测试数据
        output_dir: 输出目录
        m: 拟合数据长度
        n: 预测步长
        ar_order: AR模型阶数 (None表示自动选择)
        fit_periods: 是否拟合周期
        verbose: 是否显示详细信息
        tolerance: 评估容差
    
    返回:
        batch_history: 批量历史数据 (num_batches, m, 2)
        batch_truth: 批量真实值 (num_batches, n, 2)
        batch_predictions: 批量预测值 (num_batches, n, 2)
        eval_result: 评估结果字典
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("滑动窗口批量测试")
    print("=" * 70)
    print(f"拟合数据长度: {m}")
    print(f"预测步长: {n}")
    print(f"AR阶数: {'自动选择' if ar_order is None else ar_order}")
    print(f"拟合周期: {fit_periods}")
    
    # 创建批量测试器
    batch_tester = LSARBatchTester(
        model_class=LSARModel,
        ar_order=ar_order,
        fit_periods=fit_periods
    )
    
    # 执行批量测试
    batch_history, batch_truth, batch_predictions = batch_tester.batch_test(
        raw_history_data=raw_history_data,
        test_data=test_data,
        m=m,
        n=n,
        verbose=verbose
    )
    
    # 保存批量测试结果
    batch_tester.save_batch_results(
        batch_history=batch_history,
        batch_truth=batch_truth,
        batch_predictions=batch_predictions,
        output_dir=output_dir
    )
    
    # 评估结果
    print("\n--- 批量测试误差统计 ---")
    eval_result = ModelEvaluator.evaluate_by_features(
        batch_truth, 
        batch_predictions
    )
    ModelEvaluator.print_metrics(eval_result)
    
    # 保存评估结果
    eval_path = os.path.join(output_dir, 'batch_eval_result.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_result, f, indent=4)
    
    print(f"\n评估结果已保存到: {eval_path}")
    
    return batch_history, batch_truth, batch_predictions, eval_result


def load_batch_test_results(
    result_dir: str,
    m: int = 2000,
    n: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    """
    从目录中加载批量测试结果
    
    参数:
        result_dir: 结果保存目录
        m: 拟合数据长度
        n: 预测步长
    
    返回:
        batch_history: 批量历史数据 (num_batches, m, 2)
        batch_truth: 批量真实值 (num_batches, n, 2)
        batch_predictions: 批量预测值 (num_batches, n, 2)
        eval_result: 评估结果字典 (如果存在)
    """
    print(f"从 {result_dir} 加载批量测试结果...")
    
    # 构建文件名
    history_file = os.path.join(result_dir, f'{m}_{n}batch_history.npy')
    truth_file = os.path.join(result_dir, f'{m}_{n}batch_truth.npy')
    predictions_file = os.path.join(result_dir, f'{m}_{n}batch_predictions.npy')
    eval_file = os.path.join(result_dir, 'batch_eval_result.json')
    
    # 检查文件是否存在
    for file_path, file_name in [
        (history_file, 'batch_history'),
        (truth_file, 'batch_truth'),
        (predictions_file, 'batch_predictions')
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件: {file_path}")
    
    # 加载numpy数组
    batch_history = np.load(history_file)
    batch_truth = np.load(truth_file)
    batch_predictions = np.load(predictions_file)
    
    print(f"✓ 加载成功:")
    print(f"  - batch_history: {batch_history.shape}")
    print(f"  - batch_truth: {batch_truth.shape}")
    print(f"  - batch_predictions: {batch_predictions.shape}")
    
    # 加载评估结果(如果存在)
    eval_result = None
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            eval_result = json.load(f)
        print(f"  - 评估结果: {eval_file}")
    else:
        print(f"  - 评估结果文件不存在: {eval_file}")
    
    return batch_history, batch_truth, batch_predictions, eval_result


# ============ 使用示例 ============
if __name__ == "__main__":
    import os
    from data_pipeline import DataPipeline
    from config import DATA_CONFIG
    OUTPUT_DIR = 'data/models_baseline/ls_ar/'
    
    # 创建输出目录
    os.makedirs('data/models_baseline/ls_ar', exist_ok=True)
    
    # 加载数据
    data_pipeline = DataPipeline(DATA_CONFIG)
    _, raw_history_data, test_data = data_pipeline.splitter.time_split(
        data_pipeline.loader.load_xy_from_csv(data_pipeline.config.dataset_path), 0.0, 0.85
    )
    history_data = raw_history_data[-3650:]
    
    # print("=" * 70)
    # print("示例1: 固定周期模式 (fit_periods=False) - 单次拟合和预测")
    # print("=" * 70)
    # model1 = LSARModel(ar_order=None, fit_periods=False)
    # fit_results1 = model1.fit(history_data, verbose=True)
    # model1.save_model(OUTPUT_DIR + 'model_fixed_periods.pkl')
    # model1.save_metrics(OUTPUT_DIR + 'fit_metrics_fixed.json')
    
    # # 预测示例
    # print("\n--- 预测 30 天 ---")
    # predictions_30 = model1.predict(30)
    # print(f"预测结果形状: {predictions_30.shape}")
    # print(f"前5天预测:\n{predictions_30[:5]}")
    
    # # 带置信区间的预测
    # print("\n--- 带置信区间的预测 ---")
    # pred_mean, pred_lower, pred_upper = model1.predict_with_uncertainty(30, alpha=0.05)
    # print(f"第30天预测: x={pred_mean[29, 0]:.4f}, y={pred_mean[29, 1]:.4f}")
    # print(f"95%置信区间: x=[{pred_lower[29, 0]:.4f}, {pred_upper[29, 0]:.4f}]")
    
    # # 批量预测
    # print("\n--- 批量预测多个步长 ---")
    # predictor = LSARModelPredictor(model1)
    # batch_results = predictor.batch_predict([1, 7, 14, 30, 90])
    # for steps, pred in batch_results.items():
    #     print(f"{steps}天预测: 最后一天 x={pred[-1, 0]:.4f}, y={pred[-1, 1]:.4f}")
    
    # # 保存预测结果
    # model1.save_predictions(predictions_30, OUTPUT_DIR + 'predictions_30days.npy')
    # model1.save_predictions_csv(predictions_30, OUTPUT_DIR + 'predictions_30days.csv')
    # predictor.save_batch_predictions(batch_results, OUTPUT_DIR + 'batch_predictions/')
    
    
    print("\n" + "=" * 70)
    print("示例2: 滑动窗口批量测试")
    print("=" * 70)
    
    # ========== 执行批量测试 ==========
    batch_history, batch_truth, batch_predictions, eval_result = run_sliding_window_test(
        raw_history_data=raw_history_data,
        test_data=test_data,
        output_dir=OUTPUT_DIR + 'sliding_window_test/fit_periods/',
        m=2000,
        n=1100,
        ar_order=None,
        fit_periods=False,
        verbose=True,
        tolerance=0.063
    )
    
    print("\n" + "=" * 70)
    
    # ========== 加载已保存的结果 ==========
    print("\n示例: 加载已保存的批量测试结果")
    print("=" * 70)
    
    loaded_history, loaded_truth, loaded_predictions, loaded_eval = load_batch_test_results(
        result_dir=OUTPUT_DIR + 'sliding_window_test/fit_periods/',
        m=2000,
        n=1100
    )
    
    # # 验证加载的数据
    # print("\n验证数据完整性:")
    # print(f"  历史数据匹配: {np.allclose(batch_history, loaded_history)}")
    # print(f"  真实值匹配: {np.allclose(batch_truth, loaded_truth)}")
    # print(f"  预测值匹配: {np.allclose(batch_predictions, loaded_predictions)}")
    
    # 显示评估结果
    if loaded_eval:
        print("\n加载的评估结果:")
        ModelEvaluator.print_metrics(loaded_eval)
    else:
        print("\n--- 批量测试误差统计 ---")
        eval_result = ModelEvaluator.evaluate_by_features(
            loaded_truth, 
            loaded_predictions
        )
        ModelEvaluator.print_metrics(eval_result)
    
    # 绘制mae by day
    mae_by_days = calculate_mae_by_step(loaded_truth, loaded_predictions)
    total_mae_by_day = calculate_mae_of_dataset(mae_by_days[:100])
    mae_list = total_mae_by_day.tolist()  # 转换为Python列表
    json_str = json.dumps(mae_list)  # 转换为JSON字符串

    # 保存到文件
    with open(OUTPUT_DIR + 'sliding_window_test/fit_periods/mae_result1100_100d.json', 'w') as f:
        json.dump(mae_list, f, indent=2)
    plot_mae_by_step({"1100": total_mae_by_day})

    plot_pm(loaded_truth[2200], loaded_predictions[2200])
    
    # print("\n" + "=" * 70)
    # print("示例3: 拟合周期模式 (fit_periods=True)")
    # print("=" * 70)
    # model2 = LSARModel(ar_order=None, fit_periods=True)
    # fit_results2 = model2.fit(history_data, verbose=True)
    # model2.save_model('data/models_baseline/ls_ar/model_fitted_periods.pkl')
    # model2.save_metrics('data/models_baseline/ls_ar/fit_metrics_fitted.json')