#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
numpy 和 tensorflow 是主要的科学计算库和深度学习框架。
Sequential 是构建模型的类，Dense, Conv2D, MaxPooling2D, Flatten, Dropout 是用于构建网络层的函数。
mnist 是包含MNIST数据集的模块，to_categorical 用于将类标签转换为二进制类矩阵

'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# In[5]:


'''
reshape(-1, 28, 28, 1) 将图像数据从原始的平面数组格式转换为 CNN 所需的四维格式：样本数 x 高度 x 宽度 x 通道数。
这里图像是灰度的，所以通道数为1。
astype('float32') 将数据类型转换为 float32，用于提高处理效率。
/ 255 将像素值标准化到 0-1 范围内，有助于模型训练的稳定性和效率。
'''
# 加载并预处理数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
#to_categorical 将类别标签（从0到9的整数）转换为二进制类矩阵，用于分类。这里 10 表示总共有 10 个类别。
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[ ]:


'''
Sequential 模型是层的线性堆叠。
Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)): 第一个卷积层，有32个滤波器，每个滤波器大小为3x3，激活函数是ReLU。
input_shape=(28, 28, 1) 指定输入数据的形状。
MaxPooling2D(pool_size=(2, 2)): 第一个池化层，使用2x2的窗口进行最大池化。
Conv2D(64, (3, 3), activation='relu'): 第二个卷积层，有64个滤波器。
MaxPooling2D(pool_size=(2, 2)): 第二个池化层。
Flatten(): 展平层，将前面的多维输入一维化，用于全连接层。
Dense(128, activation='relu'): 全连接层，有128个神经元。
Dropout(0.5): Dropout层，随机关闭50%的神经元，防止过拟合。
Dense(10, activation='softmax'): 输出层，

'''
# 构建模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# In[ ]:




