#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
使用TensorFlow和Keras框架来演示如何设置一个简单的多任务学习模型，
该模型将同时对图像进行分类和回归分析。我们将使用修改后的MNIST数据集，
其中分类任务是识别手写数字，而回归任务则是预测图像的平均像素值（作为简化示例）。
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载并预处理数据
#从 TensorFlow 加载 MNIST 手写数字数据集。
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#将图像数据转换为浮点型，并进行归一化处理，使像素值位于 0 到 1 之间。
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
'''
to_categorical 函数将类别向量（整数）转换为二进制类矩阵，这对于需要分类输出的模型来说是必需的。这里的10表示有10个类别。
例如，如果一个样本属于第3类，它的类别向量将被转换为[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]。
'''
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

#计算训练集和测试集图像的平均像素值，
y_train_reg = np.mean(x_train, axis=(1, 2))
y_test_reg = np.mean(x_test, axis=(1, 2))
#作为回归任务的目标。将图像数据扩展一个维度，以适应 Keras 模型输入的要求，适用于后续的卷积操作。
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 构建多任务学习模型，定义模型输入层，指定输入图像的尺寸为 28x28x1。
inputs = Input(shape=(28, 28, 1))

# 特征提取层
#连续使用两个卷积层和最大池化层来提取图像的特征，最后使用 Flatten() 层将多维输入一维化。
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# 分类任务分支
#定义分类任务的输出层，使用 softmax 激活函数进行多分类（10 类）。
classification_output = Dense(10, activation='softmax', name='class_output')(x)

# 回归任务分支
#定义回归任务的输出层，使用 linear 激活函数输出一个连续值。
regression_output = Dense(1, activation='linear', name='reg_output')(x)

# 定义模型，指定输入和输出。
model = Model(inputs=inputs, outputs=[classification_output, regression_output])

# 编译模型，分别为两个输出指定损失函数和优化器
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'reg_output': 'mse'},
              metrics={'class_output': 'accuracy', 'reg_output': 'mse'})

# 训练模型，为两个不同的任务指定不同的损失函数和评价指标。
model.fit(x_train, {'class_output': y_train_cat, 'reg_output': y_train_reg},
          validation_data=(x_test, {'class_output': y_test_cat, 'reg_output': y_test_reg}),
          epochs=10,
          batch_size=64)

# 可视化输入和训练后的结果
def plot_results(images, true_labels, true_regress, preds):
    class_preds, reg_preds = preds
    plt.figure(figsize=(10, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Class: {np.argmax(class_preds[i])} True: {true_labels[i]}\n"
                  f"Reg: {reg_preds[i][0]:.2f} True: {true_regress[i]:.2f}")
        plt.axis('off')
    plt.show()

# 预测和可视化
predictions = model.predict(x_test[:9])
plot_results(x_test[:9], y_test[:9], y_test_reg[:9], predictions)


# In[ ]:




