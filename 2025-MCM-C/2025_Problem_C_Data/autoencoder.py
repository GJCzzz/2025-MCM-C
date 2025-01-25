import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers

# 读取数据
data = pd.read_csv('final_medal_data.csv')

# 提取特征
features = data.iloc[:, 2:].values

# 特征标准化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 设置输入和编码维度
input_dim = features_scaled.shape[1]
encoding_dim = 1  # 输出标量

# 构建模型
input_layer = Input(shape=(input_dim,))

# 增加隐藏层，增加复杂度
encoded = Dense(128, activation='relu', kernel_initializer=HeNormal())(input_layer)
encoded = Dropout(0.2)(encoded)  # Dropout层，防止过拟合
encoded = Dense(64, activation='relu')(encoded)  # 第二层隐藏层
encoded = Dropout(0.2)(encoded)  # Dropout层，防止过拟合
encoded = Dense(encoding_dim, activation='linear')(encoded)  # 最终编码层，输出标量

# 解码层（将编码的标量重构回原始输入）
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # 解码层，恢复输入维度

# 定义自编码器和编码器模型
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# 编译模型，使用更好的损失函数
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')  # 使用 MAE 损失函数

# 训练模型

# 使用学习率下降策略
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
autoencoder.fit(features_scaled, features_scaled, epochs=100, batch_size=256, shuffle=True, callbacks=[lr_scheduler])

# 获取编码后的标量特征
encoded_data = encoder.predict(features_scaled)

# 将编码结果添加到原数据中
data['EncodedFeature'] = encoded_data

# 将结果保存为 CSV 文件
data.to_csv('encoded_data.csv', index=False)
print("模型训练完成，已将编码结果保存为 'encoded_data.csv'。")