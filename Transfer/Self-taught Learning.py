#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
在这个示例中，
我们将使用 MNIST 数据集的未标记部分和 Fashion MNIST 数据集的有标记部分来展示 self-taught learning。
这个任务将演示如何使用自编码器从未标记的 MNIST 数据中学习特征，
然后使用这些特征来改进在 Fashion MNIST 上的分类任务。
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical

# 加载数据
'''
加载 MNIST 和 Fashion MNIST 数据集。
MNIST 用作未标记数据训练自编码器，Fashion MNIST 用作有标记数据训练分类器。
归一化图像数据到 [0, 1] 范围，将图像的形状扩展一个维度以适应 Keras 需求。
对 Fashion MNIST 的标签进行独热编码处理，以便进行多分类。
to_categorical 函数将类别向量（整数）转换为二进制类矩阵，这对于需要分类输出的模型来说是必需的。这里的10表示有10个类别。
例如，如果一个样本属于第3类，它的类别向量将被转换为[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]。
'''
def load_data():
    #加载未标记的 MNIST 数据用于自编码器训练，加载有标记的 Fashion MNIST 数据用于分类任务。
    (x_train_mnist, _), (x_test_mnist, _) = mnist.load_data()
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()

    # 使用未标记的MNIST数据作为自编码器的输入
    x_train_mnist = x_train_mnist.astype('float32') / 255.
    x_train_mnist = np.expand_dims(x_train_mnist, -1)

    # 使用Fashion MNIST的有标记数据进行分类任务
    x_train_fmnist = x_train_fmnist.astype('float32') / 255.
    x_train_fmnist = np.expand_dims(x_train_fmnist, -1)
    y_train_fmnist = to_categorical(y_train_fmnist, 10)

    return x_train_mnist, x_train_fmnist, y_train_fmnist

x_train_mnist, x_train_fmnist, y_train_fmnist = load_data()

# 构建自编码器模型
input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自编码器
'''
epochs因为跑的久所以只设定10，你可以设50或更大。或使用早停的方法
'''
autoencoder.fit(x_train_mnist, x_train_mnist, epochs=10, batch_size=256, shuffle=True)

# 使用自编码器的编码部分作为特征提取器
feature_extractor = Model(inputs=autoencoder.input, outputs=encoded)

# 从Fashion MNIST数据中提取特征
fashion_features = feature_extractor.predict(x_train_fmnist)

# 构建分类模型
input_features = Input(shape=fashion_features.shape[1:])
flattened_features = Flatten()(input_features)
classifier_output = Dense(10, activation='softmax')(flattened_features)
classifier = Model(inputs=input_features, outputs=classifier_output)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练分类模型
classifier.fit(fashion_features, y_train_fmnist, epochs=30, batch_size=256)

# 可视化自编码器的输入和输出
decoded_imgs = autoencoder.predict(x_train_mnist[:10])
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_mnist[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 显示重构图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




