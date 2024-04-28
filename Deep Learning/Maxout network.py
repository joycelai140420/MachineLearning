#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# In[6]:


def load_data():
    # 加载数据集并处理
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()


# In[7]:


def maxout(inputs, num_units, axis=-1):
    # Maxout激活函数实现
    '''
    tf.reduce_max 函数用于计算张量沿指定轴的最大值。
    tf.reduce_max: 这个操作接受输入张量，并沿指定的轴计算其最大值。
    axis: 在 Maxout 网络中，我们需要沿每个 Maxout 单元的子单元维度 (sub_units) 计算最大值，即对每个 num_units 组里的元素求最大。
          通常这个轴是 -1（最后一个维度），意味着我们希望沿每个 Maxout 单元中的子单元计算最大值。
    
    tf.reshape 函数用于改变 inputs 张量的形状，以便于执行 Maxout 操作。
    inputs: 假设是一个 [batch_size, features] 形状的张量。
    num_units: Maxout 单元的数量。这是一个超参数，你可以根据网络设计自行设定。
    inputs.shape[1]//num_units: 这部分计算每个 Maxout 单元包含的子单元数。
                                如果 inputs 的第二维（特征维）是 2560，并且 num_units 是 512，那么每个 Maxout 单元将有 5 个子单元。
    
    结果是将输入张量 inputs 重塑成一个三维张量，形状为 [batch_size, num_units, sub_units]，其中：
    batch_size 是每批数据的大小。
    num_units 是 Maxout 单元的数量。
    sub_units 是每个 Maxout 单元中的子单元数。
    '''

    outputs = tf.reduce_max(tf.reshape(inputs, (-1, num_units, inputs.shape[1]//num_units)), axis=axis)
    return outputs


# In[8]:


model = Sequential()
# 输入层到第一层 Maxout, 假设我们有 512 个单元，每个 Maxout 单元有 5 个子单元
model.add(Dense(units=512 * 5, input_dim=28*28))  # 乘以 5，因为每个 Maxout 单元将从 5 个中取最大值
model.add(Reshape((512, 5)))  # 重塑输出以便于取最大值
#每个 Maxout 单元从其 5 个子单元中选择最大值，使用 Lambda 层实现。
model.add(Lambda(lambda x: tf.reduce_max(x, axis=-1), output_shape=(512,)))  # Maxout 操作
model.add(Dropout(0.5))

# 第二层 Maxout
model.add(Dense(512 * 5))  # 再次，512 Maxout 单元，每个有 5 个子单元
model.add(Reshape((512, 5)))
#每个 Maxout 单元从其 5 个子单元中选择最大值，使用 Lambda 层实现。
model.add(Lambda(lambda x: tf.reduce_max(x, axis=-1), output_shape=(512,)))
model.add(Dropout(0.5))

# 输出层
model.add(Dense(10, activation='softmax'))  # 做 10 类分类

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#使用早停来避免过拟合，监控验证集损失，并在连续三次迭代无改善时停止训练。
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train, y_train, epochs=22, batch_size=100, validation_split=0.1, callbacks=[early_stopping])

train_loss, train_accuracy = model.evaluate(x_train, y_train)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"\nTraining Loss: {train_loss}, Training Accuracy: {train_accuracy}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# ## 
