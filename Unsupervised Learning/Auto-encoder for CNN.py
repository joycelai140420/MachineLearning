#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
示例展示了如何使用卷积自动编码器进行图像的特征学习和重构，这种技术广泛应用于图像处理、压缩和噪声消除等领域。
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# 加载数据
(x_train, _), (x_test, _) = mnist.load_data()

# 数据预处理
#使用 MNIST 数据集，标准化图像数据，并调整尺寸以适应网络输入。
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # 适应卷积层的输入要求
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 构建自动编码器
'''
使用 Input 定义输入层。
Conv2D 和 MaxPooling2D 用于创建编码器部分，逐渐减小空间维度，增加特征深度。
UpSampling2D 和 Conv2D 构建解码器部分，逐步恢复图像尺寸。
'''
#编码器部分
input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
#解码器部分
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

'''
编译和训练模型：
使用 Adam 优化器和二元交叉熵损失函数编译模型。
在训练数据上训练模型，并使用测试数据进行验证。
'''
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自动编码器,跑太久我将epochs=10设定比较小
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 使用自动编码器进行图像重构
decoded_imgs = autoencoder.predict(x_test)

# 可视化原始和重构后的图像
n = 10  # 展示几个数字
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 显示重构后的图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




