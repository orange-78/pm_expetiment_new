import numpy as np
from scipy.optimize import least_squares
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
import pickle
import json
from typing import Tuple, Dict, Optional


class LSARModel:
    """
    LS+AR极移预测模型
    LS: 最小二乘拟合 (趋势 + 钱德勒摆动 + 年周期)
    AR: 自回归模型 (对残差建模)
    """
    
    def __init__(self, ar_order: Optional[int] = None):
        """
        初始化模型
        
        Parameters:
        -----------
        ar_order : int, optional
            AR模型阶数，如果为None则自动选择
        """
        self.ar_order = ar_order
        
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
        PM = a0 + a1*t + a2*cos(2πt/P_C + φ_C) + a3*cos(2πt/P_A + φ_A)
        
        Parameters:
        -----------
        t : np.ndarray
            时间序列
        params : np.ndarray
            参数 [a0, a1, a2, φ_C, a3, φ_A]
            
        Returns:
        --------
        np.ndarray
            模型预测值
        """
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
        
        initial_params = np.array([a0_init, a1_init, a2_init, phi_c_init, a3_init, phi_a_init])
        
        # 非线性最小二乘拟合
        result = least_squares(
            self._ls_residuals,
            initial_params,
            args=(t, y),
            method='lm',
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
        x_fitted = self._ls_model(t, self.ls_params_x) + self.ar_fit_x.fittedvalues
        y_fitted = self._ls_model(t, self.ls_params_y) + self.ar_fit_y.fittedvalues
        
        # 确保长度一致（AR模型会损失前几个点）
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
            'ar_order_x': ar_order_x,
            'ar_order_y': ar_order_y,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'aic_x': self.ar_fit_x.aic,
            'aic_y': self.ar_fit_y.aic,
            'ls_params_x': self.ls_params_x.tolist(),
            'ls_params_y': self.ls_params_y.tolist(),
        }
        
        if verbose:
            print("\n===== 拟合结果 =====")
            print(f"X方向: R² = {r2_x:.6f}, RMSE = {rmse_x:.6f} mas")
            print(f"Y方向: R² = {r2_y:.6f}, RMSE = {rmse_y:.6f} mas")
            print(f"X方向LS参数: {self.ls_params_x}")
            print(f"Y方向LS参数: {self.ls_params_y}")
        
        return self.fit_metrics
    
    def save_model(self, filepath: str):
        """保存模型到文件"""
        model_data = {
            'ar_order': self.ar_order,
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


# ============ 使用示例 ============
if __name__ == "__main__":
    from data_pipeline import DataLoader, DataSplitter
    
    # 加载数据
    _, raw_history_data, test_data = DataSplitter.time_split(
        DataLoader.load_xy_from_csv, 0.0, 0.85
    )
    history_data = raw_history_data[-3650:]
    
    # 创建并拟合模型
    model = LSARModel(ar_order=None)  # 自动选择AR阶数
    fit_results = model.fit(history_data, verbose=True)
    
    # 保存模型和指标
    model.save_model('lsar_model.pkl')
    model.save_metrics('lsar_fit_metrics.json')