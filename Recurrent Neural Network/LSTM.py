#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 模拟加载数据函数
def load_data():
    # 这里使用随机数据模拟股票价格
    np.random.seed(0)
    timeseries = np.sin(0.1 * np.arange(200)) + np.random.normal(0, 0.1, 200)
    return timeseries

# 准备数据
#data: 输入的一维时间序列数组。
#n_steps: 每个训练输入数据包含的连续时间步数量。
#这里就是记录课程内容x^t ,x^t+1
def prepare_data(data, n_steps):
    X, y = [], []
    '''
    这个循环遍历时间序列数据，i 从 0 开始到 len(data) - n_steps。
    这确保了每个输入序列都可以从数据中提取出 n_steps 长度的数据，
    并且对于每个序列都有一个对应的目标值可以访问。
    
    data[i:i+n_steps]: 从位置 i 开始，提取长度为 n_steps 的子序列。
    这个子序列作为输入数据 X 的一部分。
    
    data[i+n_steps]: 提取位于输入序列之后的单个数据点。
    这是给定前 n_steps 个值后，模型需要预测的下一个值，作为输出数据 y 的一部分。
    '''
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# 加载数据
data = load_data()
n_steps = 10
X, y = prepare_data(data, n_steps)

# 数据维度调整以适应 LSTM 输入 [samples, time_steps, features]
'''
使用 LSTM (Long Short-Term Memory) 网络处理时间序列数据时，输入数据通常需要有特定的形状。这个形状通常是三维的，包括以下几个维度：

样本数量 (samples)：这是数据集中时间序列的总数或批次中的序列数量。
时间步长 (time_steps)：这表示每个序列的长度，即序列中的观测点数。
特征数量 (features)：每个时间步中观测到的特征数量。
例如:
X = X.reshape((X.shape[0], X.shape[1], 1))

X.shape[0]：这代表 X 中的总样本数，即时间序列的数量。
X.shape[1]：这代表每个时间序列的长度，也就是每个样本的时间步长。
1：这代表每个时间步中的特征数量。在这个例子中，每个时间步只有一个特征
（可能是单一的时间序列数据，如股价或其他度量值）。
因此，我们需要增加一个维度，使其成为 (样本数, 时间步长, 特征数) 的形式，
这里的特征数是 1。
LSTM 层期望输入的形状是 (samples, time_steps, features)，
这样它才能正确地处理输入的时间序列数据。
增加一个维度是因为即使每个时间步只有一个特征，Keras 的 LSTM 层也需要明确地知道特征的数量。
这使得模型能够灵活处理可能来自多变量时间序列的多个特征的情况。
如果你的时间序列数据包含多个特征，比如同时记录了温度、湿度和压力，
那么在重塑数据时，最后一个维度将不是 1 而是特征的实际数量，
例如:
X = X.reshape((X.shape[0], X.shape[1], 3))  # 如果每个时间步有三个特征
'''
X = X.reshape((X.shape[0], X.shape[1], 1))

# 构建 LSTM 模型
'''
使用 Sequential 模型，添加一个 LSTM 层和一个输出层（Dense）。
LSTM 层中的单元数为 50，这是可以调整的超参数。
50就是50个z,zi,zf，zo，a

使用 Adam 优化器和均方误差（MSE）损失函数来训练模型。
'''
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(lr=0.01), loss='mse')

# 训练模型
model.fit(X, y, epochs=200, verbose=0)

# 进行预测
y_pred = model.predict(X)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Time Series')
plt.plot(np.arange(n_steps, len(y_pred)+n_steps), y_pred, label='Predicted Time Series')
plt.title('Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[ ]:




