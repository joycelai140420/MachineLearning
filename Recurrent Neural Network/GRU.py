#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# 生成模拟天气数据
def generate_temperature_data(num_days, base_temp=10, temp_variation=15):
    np.random.seed(42)  # 设置随机种子以确保结果的可重现性。
    hours = np.arange(num_days * 24)#生成一个整数序列，表示一年中的每个小时（假设每天24小时，num_days天）。
    '''
     使用正弦函数模拟日内温度波动。2 * np.pi * hours / 24 将时间转换为正弦函数的输入，确保每24小时完成一个周期。
     temp_variation 控制温度波动的幅度。
    '''
    daily_temp_variation = np.sin(2 * np.pi * hours / 24) * temp_variation
    '''
    生成高斯噪声，模拟天气数据中的随机波动，scale=2 控制噪声的标准差。
    '''
    random_noise = np.random.normal(size=num_days * 24, scale=2)
    '''
    最终的温度数据由基础温度、日变化和随机噪声组成。
    '''
    temperatures = base_temp + daily_temp_variation + random_noise
    return temperatures

# 准备数据
def prepare_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):#每次滑动窗口n_steps小时，这个窗口定义了用来预测下一小时温度的时间序列长度。
        X.append(data[i:i+n_steps])#将每个长度为n_steps的时间序列添加到输入列表X。
        y.append(data[i+n_steps])#将每个时间序列后的一个数据点作为目标值添加到输出列表y。
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # 重新塑形X以满足GRU输入的要求。GRU需要三维输入：[样本数，时间步长，特征数]。
    return X, y

n_steps = 24  # use 24 hours of data to predict the next hour
data = generate_temperature_data(365)  # simulate a year of hourly temperatures

X, y = prepare_sequences(data, n_steps)

# Build the model
model = Sequential([
    GRU(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, verbose=1)

# Make predictions
predicted = model.predict(X)

# Visualize the data
plt.figure(figsize=(15, 5))
plt.plot(data, label='Actual Temperature')
plt.plot(np.arange(n_steps, len(predicted) + n_steps), predicted.ravel(), label='Predicted Temperature', color='red')
plt.title('Temperature Prediction')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()


# In[ ]:




