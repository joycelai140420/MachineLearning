#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
使用 TensorFlow 和 Keras 构建一个简单的自动编码器（Auto-encoder），
并使用 MNIST 手写数字数据集进行训练。
这个示例将展示如何建立模型，训练它，并对训练后的结果进行可视化。

'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# 加载 MNIST 数据集
#这里只关心图像数据 (x_train 和 x_test)，而不加载标签数据（用 _ 忽略）。
(x_train, _), (x_test, _) = mnist.load_data()

# 数据预处理
'''
在前面都有说明,/255是转换成黑白，黑=1
将图像数据类型转换为浮点数并除以 255，实现归一化（将像素值从 [0, 255] 转换到 [0, 1]）。
之后将每个 28x28 的图像重新塑形为 784 维的向量（因为全连接层需要一维输入）。
'''

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 设置编码器和解码器的大小,意味着输入图像将被压缩从 784 维降到 32 维。
encoding_dim = 32  # 编码的大小

# 定义输入层,指定输入尺寸为 784（MNIST图像的维数）。
input_img = Input(shape=(784,))

# 编码层是一个全连接（Dense）层，使用 relu 激活函数。它将输入的维度从 784 降至 32。
encoded = Dense(encoding_dim, activation='relu')(input_img)

# 解码层又是一个全连接层，使用 sigmoid 激活函数将维度从 32 扩展回 784。
decoded = Dense(784, activation='sigmoid')(encoded)

# 构建自动编码器模型,实例化自动编码器模型，指定输入和输出层，创建一个从输入到解码输出的模型。
autoencoder = Model(input_img, decoded)

# 构建编码器模型，单独创建编码器部分的模型，用于后续生成编码后的数据。
encoder = Model(input_img, encoded)

# 构建解码器模型
'''
单独创建解码器部分的模型。首先定义一个新输入层，专门用于接受编码（潜在空间向量）。
然后获取自动编码器中的最后一层autoencoder.layers[-1]（解码层），并使用它构建解码模型。
'''
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# 编译模型
'''
Adam 在实践中被证明在多种条件下都表现良好，尤其是处理大规模数据或参数的场景中。
它相比其他基本优化器（如SGD）通常能更快收敛。
binary_crossentropy 如果输出层是多分类的，可能会选择分类交叉熵损失函数
'''
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自动编码器
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 使用测试集的编码和解码图像
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# 可视化结果
n = 10  # 展示多少数字
plt.figure(figsize=(20, 4))
for i in range(n):
    # 原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 重建后的图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




