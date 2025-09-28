from tf_singleton import tf
import numpy as np
import matplotlib.pyplot as plt

# 示例调用
# (X_train, y_train), (X_val, y_val), (X_test, y_test), scalers, (train_raw, val_raw, test_raw) = prepare_datasets(...)
# T_test, p_test = get_true_pred_sequences("best_model_weights.h5", X_test, y_test)

def plot_pm(T_test, p_test, i):
    """
    可视化第 i 条数据的 PMX 和 PMY 真值与预测值对比
    
    参数：
        T_test: np.ndarray, shape (n, lookback+steps, 2) 真实序列
        p_test: np.ndarray, shape (n, lookback+steps, 2) 预测序列
        i: int, 要绘制的样本索引
    """
    true_seq = T_test[i]
    pred_seq = p_test[i]
    
    # 时间轴
    timesteps = np.arange(true_seq.shape[0])
    
    # 提取 PMX 和 PMY
    pmx_true, pmy_true = true_seq[:, 0], true_seq[:, 1]
    pmx_pred, pmy_pred = pred_seq[:, 0], pred_seq[:, 1]
    
    # 绘图
    fig, axes= plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # PMX
    axes[0].plot(timesteps, pmx_true, label="True PMX", color="blue")
    axes[0].plot(timesteps, pmx_pred, label="Pred PMX", color="red", linestyle="--")
    axes[0].set_ylabel("PMX")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # PMY
    axes[1].plot(timesteps, pmy_true, label="True PMY", color="blue")
    axes[1].plot(timesteps, pmy_pred, label="Pred PMY", color="red", linestyle="--")
    axes[1].set_ylabel("PMY")
    axes[1].set_xlabel("Time step")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# 使用示例
# plot_pmx_pmy(T_test, p_test, i=0)


