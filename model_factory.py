"""
模型工厂 - model_factory.py
"""

from tf_singleton import tf
from tensorflow.keras.layers import ( # type: ignore
    Input, LSTM, Dropout, MultiHeadAttention, Add, LayerNormalization, Dense, Reshape
)
from tensorflow.keras.models import Model # type: ignore
from config import ModelConfig


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_lstm_attention_model(lookback: int, steps: int, config: ModelConfig) -> Model:
        """创建LSTM+注意力模型"""
        
        # 输入层
        input_layer = Input(shape=(lookback, config.num_features), name='input')

        # 第一层 LSTM + Dropout
        x = LSTM(config.lstm0, return_sequences=True, name='lstm_1')(input_layer)
        x = Dropout(config.dropout0, name='dropout_1')(x)

        # 第二层 LSTM + Dropout
        x = LSTM(config.lstm1, return_sequences=True, name='lstm_2')(x)
        x = Dropout(config.dropout1, name='dropout_2')(x)

        # 多头注意力层
        attn_output = MultiHeadAttention(
            num_heads=config.attnhead, 
            key_dim=config.attndim,
            name='multi_head_attention'
        )(x, x)
        attn_output = Dropout(config.dropout2, name='attention_dropout')(attn_output)

        # 残差连接 + Layer Normalization
        x = Add(name='residual_connection')([x, attn_output])
        x = LayerNormalization(name='layer_norm')(x)

        # 第三层 LSTM
        x = LSTM(config.lstm2, return_sequences=False, name='lstm_3')(x)

        # 输出层
        x = Dense(steps * config.num_features, name='dense_output')(x)
        output = Reshape((steps, config.num_features), name='reshape_output')(x)

        # 构建模型
        model = Model(inputs=input_layer, outputs=output, name='lstm_attention_model')
        
        return model
    
    @staticmethod
    def create_simple_lstm_model(lookback: int, steps: int, config: ModelConfig) -> Model:
        """创建简单LSTM模型（用于对比实验）- 结构与LSTM+Attention模型一致但不含注意力层"""
        
        # 输入层
        input_layer = Input(shape=(lookback, config.num_features), name='input')

        # 第一层 LSTM + Dropout
        x = LSTM(config.lstm0, return_sequences=True, name='lstm_1')(input_layer)
        x = Dropout(config.dropout0, name='dropout_1')(x)

        # 第二层 LSTM + Dropout
        x = LSTM(config.lstm1, return_sequences=True, name='lstm_2')(x)
        x = Dropout(config.dropout1, name='dropout_2')(x)

        # 跳过注意力层和残差连接
        # 直接进行 Layer Normalization（保持结构一致性）
        x = LayerNormalization(name='layer_norm')(x)

        # 第三层 LSTM
        x = LSTM(config.lstm2, return_sequences=False, name='lstm_3')(x)

        # 输出层
        x = Dense(steps * config.num_features, name='dense_output')(x)
        output = Reshape((steps, config.num_features), name='reshape_output')(x)

        # 构建模型
        model = Model(inputs=input_layer, outputs=output, name='simple_lstm_model')
        
        return model
    
    @staticmethod
    def create_transformer_model(lookback: int, steps: int, config: ModelConfig) -> Model:
        """创建Transformer模型（可选的新模型类型）"""
        
        input_layer = Input(shape=(lookback, config.num_features), name='input')
        
        # 多头注意力
        attn1 = MultiHeadAttention(
            num_heads=config.attnhead, 
            key_dim=config.attndim,
            name='attention_1'
        )(input_layer, input_layer)
        attn1 = Dropout(config.dropout0, name='dropout_attn1')(attn1)
        
        # 残差连接和归一化
        x = Add(name='residual_1')([input_layer, attn1])
        x = LayerNormalization(name='norm_1')(x)
        
        # 第二个注意力层
        attn2 = MultiHeadAttention(
            num_heads=config.attnhead, 
            key_dim=config.attndim,
            name='attention_2'
        )(x, x)
        attn2 = Dropout(config.dropout1, name='dropout_attn2')(attn2)
        
        # 残差连接和归一化
        x = Add(name='residual_2')([x, attn2])
        x = LayerNormalization(name='norm_2')(x)
        
        # 全局平均池化
        x = tf.keras.layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # 输出层
        x = Dense(config.lstm2, activation='relu', name='dense_1')(x)
        x = Dropout(config.dropout2, name='dropout_final')(x)
        x = Dense(steps * config.num_features, name='dense_output')(x)
        output = Reshape((steps, config.num_features), name='reshape_output')(x)
        
        model = Model(inputs=input_layer, outputs=output, name='transformer_model')
        
        return model
    
    @classmethod
    def get_model(cls, model_type: str, lookback: int, steps: int, config: ModelConfig) -> Model:
        """根据类型获取模型"""
        
        model_map = {
            'lstm_attention': cls.create_lstm_attention_model,
            'simple_lstm': cls.create_simple_lstm_model,
            'transformer': cls.create_transformer_model,
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_map.keys())}")
        
        return model_map[model_type](lookback, steps, config)