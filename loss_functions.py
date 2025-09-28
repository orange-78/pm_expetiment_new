"""
损失函数模块 - loss_functions.py
"""

from tf_singleton import tf
from tensorflow.keras import backend as K # type: ignore


class LossFunctions:
    """损失函数集合"""

    # -------------------------
    # 基础频域损失
    # -------------------------
    @staticmethod
    def fft_loss(y_true, y_pred):
        """频域损失：对序列做FFT后比较幅度谱"""
        true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))

        true_mag = tf.abs(true_fft)
        pred_mag = tf.abs(pred_fft)

        return tf.reduce_mean(tf.square(true_mag - pred_mag))

    @staticmethod
    def phase_loss(y_true, y_pred):
        """相位损失：比较FFT相位差"""
        true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))

        true_phase = tf.math.angle(true_fft)
        pred_phase = tf.math.angle(pred_fft)

        return tf.reduce_mean(tf.square(true_phase - pred_phase))

    # -------------------------
    # 组合损失
    # -------------------------
    @staticmethod
    def mae_freq_loss(alpha=1.0, beta=0.2, gamma=0.1):
        """组合损失：时域 MAE + 频域差异 + 相位差异"""
        def loss(y_true, y_pred):
            time_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            freq_loss = LossFunctions.fft_loss(y_true, y_pred)
            phase_loss = LossFunctions.phase_loss(y_true, y_pred)
            return alpha * time_loss + beta * freq_loss + gamma * phase_loss
        return loss

    # -------------------------
    # 相关系数指标
    # -------------------------
    @staticmethod
    def corr_metric(y_true, y_pred):
        """整体 Pearson 相关系数"""
        x = y_true - K.mean(y_true, axis=-1, keepdims=True)
        y = y_pred - K.mean(y_pred, axis=-1, keepdims=True)

        corr = K.sum(x * y, axis=-1) / (
            K.sqrt(K.sum(K.square(x), axis=-1)) * K.sqrt(K.sum(K.square(y), axis=-1)) + K.epsilon()
        )
        return K.mean(corr)

    @staticmethod
    def feature_wise_corr_metric(y_true, y_pred):
        """按特征维度分别计算相关系数"""
        x = y_true - K.mean(y_true, axis=0, keepdims=True)
        y = y_pred - K.mean(y_pred, axis=0, keepdims=True)

        numerator = K.sum(x * y, axis=0)
        denominator = K.sqrt(K.sum(K.square(x), axis=0)) * K.sqrt(K.sum(K.square(y), axis=0)) + K.epsilon()
        corr = numerator / denominator
        return K.mean(corr)

    # -------------------------
    # 相关系数组合损失
    # -------------------------
    @staticmethod
    def mse_corr_loss(alpha=5e-4):
        """MSE + 相关系数损失"""
        def loss(y_true, y_pred):
            mse = K.mean(K.square(y_true - y_pred))
            corr = LossFunctions.corr_metric(y_true, y_pred)
            return mse + alpha * (1 - corr)
        return loss

    @staticmethod
    def mae_corr_loss(alpha=5e-4):
        """MAE + 相关系数损失"""
        def loss(y_true, y_pred):
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            corr = LossFunctions.corr_metric(y_true, y_pred)
            return mae + alpha * (1 - corr)
        return loss

    @staticmethod
    def huber_corr_loss(delta=1.0, alpha=5e-4):
        """Huber + 相关系数损失"""
        def loss(y_true, y_pred):
            error = y_true - y_pred
            abs_error = tf.abs(error)
            huber = tf.where(
                abs_error <= delta,
                0.5 * tf.square(error),
                delta * (abs_error - 0.5 * delta)
            )
            huber_loss = tf.reduce_mean(huber)
            corr = LossFunctions.corr_metric(y_true, y_pred)
            return huber_loss + alpha * (1 - corr)
        return loss

    # -------------------------
    # Focal MSE 损失
    # -------------------------
    @staticmethod
    def focal_mse_loss(gamma=2.0):
        """Focal MSE：增强对难样本的关注"""
        def loss(y_true, y_pred):
            mse = tf.square(y_true - y_pred)
            weight = tf.pow(tf.abs(y_true - y_pred), gamma)
            return tf.reduce_mean(weight * mse)
        return loss

    # -------------------------
    # 分位数损失
    # -------------------------
    @staticmethod
    def quantile_loss(q=0.5):
        """分位数损失"""
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
        return loss

    # -------------------------
    # 工厂方法
    # -------------------------
    @classmethod
    def get_loss_function(cls, loss_name: str, **kwargs):
        """根据名称获取损失函数"""
        loss_map = {
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae_freq': cls.mae_freq_loss,
            'mse-corr': cls.mse_corr_loss,
            'mae-corr': cls.mae_corr_loss,
            'huber-corr': cls.huber_corr_loss,
            'focal-mse': cls.focal_mse_loss,
            'quantile': cls.quantile_loss,
            'fft': cls.fft_loss,
            'phase': cls.phase_loss,
        }

        if loss_name not in loss_map:
            raise ValueError(f"Unknown loss function: {loss_name}")

        loss_func = loss_map[loss_name]

        if callable(loss_func) and loss_name in [
            'mae_freq', 'mse-corr', 'mae-corr', 'huber-corr', 'focal-mse', 'quantile'
        ]:
            return loss_func(**kwargs)
        else:
            return loss_func

    @classmethod
    def get_metrics(cls, metrics_name: str, **kwargs):
        """根据名称获取评估指标"""
        metric_map = {
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'mse': tf.keras.metrics.MeanSquaredError(),
            'corr': cls.corr_metric,
            'feature-corr': cls.feature_wise_corr_metric,
        }

        # 如果指定 metrics="loss"，则根据 loss_name 自动匹配指标
        if metrics_name == 'loss':
            loss_name = kwargs.get('loss_name', 'mae')

            # loss → 推荐的 metrics 组合
            loss_to_metrics = {
                'mae': ['mae'],
                'mse': ['mse'],
                'mae_freq': ['mae', 'fft', 'phase'],
                'mse-corr': ['mse', 'corr'],
                'mae-corr': ['mae', 'corr'],
                'huber-corr': ['mae', 'corr'],   # Huber ~ MAE
                'focal-mse': ['mse'],
                'quantile': ['mae'],             # 常用回归误差度量
                'fft': ['fft'],
                'phase': ['phase'],
            }

            if loss_name not in loss_to_metrics:
                raise ValueError(f"Unknown loss name for metrics matching: {loss_name}")

            selected_metrics = []
            for m in loss_to_metrics[loss_name]:
                if m in metric_map:
                    selected_metrics.append(metric_map[m])
                elif m == 'fft':
                    selected_metrics.append(cls.fft_loss)
                elif m == 'phase':
                    selected_metrics.append(cls.phase_loss)
            return selected_metrics

        # 常规定义
        if metrics_name not in metric_map:
            raise ValueError(f"Unknown metric: {metrics_name}")
        return metric_map[metrics_name]

    @staticmethod
    def list_available_losses():
        """列出可用的损失函数"""
        return [
            'mae', 'mse', 'mae_freq', 'mse-corr', 'mae-corr',
            'huber-corr', 'focal-mse', 'quantile', 'fft', 'phase'
        ]

    @staticmethod
    def list_available_metrics():
        """列出可用的评估指标"""
        return ['mae', 'mse', 'corr', 'feature-corr']
