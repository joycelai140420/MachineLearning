#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# 生成序列数据
def generate_data(timesteps, noise=0.1):
    x = np.arange(timesteps)
    '''
    生成一个基于正弦波形的时间序列数据，并向其添加一定量的高斯（正态分布）噪声。
    x 是一个数组，包含了时间步序列，例如 np.arange(timesteps)。这里 timesteps 是时间序列的总长度，x 通常是一个从 0 到 timesteps-1 的整数序列。
    0.1 * x：将 x 的每个值乘以 0.1。这个缩放因子（0.1）决定了正弦波的频率。乘以 0.1 是为了减缓正弦波的周期变化，使得在 x 的整个范围内，波形变化更为缓慢。
    np.sin(0.1 * x)：计算缩放后 x 数值的正弦值。正弦函数是周期性函数，常用于生成周期性波动的模型。这里产生的是一个标准的正弦波形，周期取决于乘以 x 的系数。
    np.random.normal(scale=noise, size=timesteps)：生成一个高斯噪声数组，其长度等于 timesteps，scale 参数控制噪声的标准差（即噪声的强度或大小）。在这个表达式中，noise 是一个预先定义的数值，它决定了噪声的变异程度。
    +：将正弦波形和噪声数组相加。这意味着每个时刻的正弦值都会加上一个随机生成的噪声值，从而模拟现实世界中数据的不完美性和随机性。
    这种生成数据的方式可以用于测试和开发处理时间序列数据的算法，如预测算法、趋势分析等。添加噪声是为了模拟真实环境中可能遇到的数据干扰，提高模型在处理真实世界数据时的鲁棒性和适用性。
    '''
    y = np.sin(0.1 * x) + np.random.normal(scale=noise, size=timesteps)
    return y

# 准备输入输出对，用于训练 RNN
def prepare_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for RNN [samples, timesteps, features]
    return X, y

# 参数设置
timesteps = 400  # 总时间步
n_steps = 10     # 单个输入序列的长度

# 数据生成
data = generate_data(timesteps)

# 数据准备
X, y = prepare_sequences(data, n_steps)

# 构建模型
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(lr=0.005), loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=200, verbose=0)

# 进行预测
x_input = data[-n_steps:]  # 使用最后 n_steps 数据作为输入
x_input = x_input.reshape((1, n_steps, 1))
yhat = model.predict(x_input, verbose=0)
print(f"Predicted Next Value: {yhat[0][0]}")

# 可视化数据和预测
plt.figure(figsize=(10, 6))
plt.plot(data, label='Given Data')
plt.scatter(timesteps, yhat, color='red', label='Predicted Next Value', zorder=5)
plt.title('Time Series Prediction with SimpleRNN')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[ ]:




