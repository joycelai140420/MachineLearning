#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
下面是一个完整的域对抗训练（Domain-Adversarial Training）的示例
使用 TensorFlow 和 Keras 框架来实现多任务学习，其中一个任务是分类，另一个任务是域分类，
用于区分两个不同的数据集。
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 加载并预处理数据
'''
 MNIST 和 Fashion MNIST 数据集是不一样的维度，所以对图像进行了归一化和维度调整，使其适用于后续的卷积网络处理。还对标签进行了独热编码，以适配模型输出层的预期格式。
构建模型，以防止后面合并的时候出错

'''
def load_data():
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()

    # 归一化并调整维度
    x_train_mnist = np.expand_dims(x_train_mnist.astype('float32') / 255., -1)
    x_test_mnist = np.expand_dims(x_test_mnist.astype('float32') / 255., -1)
    x_train_fmnist = np.expand_dims(x_train_fmnist.astype('float32') / 255., -1)
    x_test_fmnist = np.expand_dims(x_test_fmnist.astype('float32') / 255., -1)

    # 独热编码
    y_train_mnist = to_categorical(y_train_mnist, 10)
    y_test_mnist = to_categorical(y_test_mnist, 10)
    y_train_fmnist = to_categorical(y_train_fmnist, 10)
    y_test_fmnist = to_categorical(y_test_fmnist, 10)

    return (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,
            x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist)

# 构建多任务学习模型
'''
此函数定义了一个多任务学习模型，使用两个输出：一个用于分类（区分0-9的数字），一个用于域分类（判断是MNIST还是Fashion MNIST）。
模型使用卷积层来提取特征，然后用这些特征进行两种不同的预测任务。
'''
def build_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    features = Dense(128, activation='relu')(x)
    
    # 分类输出
    classification_output = Dense(10, activation='softmax', name='classification')(features)
    # 域分类输出
    domain_classifier = Dense(2, activation='softmax', name='domain_classification')(features)

    model = Model(inputs=inputs, outputs=[classification_output, domain_classifier])
    return model

# 可视化函数
def plot_results(images, labels, domain_labels, preds):
    plt.figure(figsize=(12, 8))
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Class: {np.argmax(labels[i])}\nDomain: {np.argmax(domain_labels[i])}")
        plt.axis('off')
    plt.show()

# 加载数据
data = load_data()
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist, x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist = data

# 合并数据集
x_combined = np.concatenate([x_train_mnist, x_train_fmnist])
y_combined_class = np.concatenate([y_train_mnist, y_train_fmnist])
domain_labels = np.concatenate([np.zeros((len(x_train_mnist), 2)), np.ones((len(x_train_fmnist), 2))])

# 构建并编译模型
model = build_model()
model.compile(optimizer='adam',
              loss={'classification': 'categorical_crossentropy', 'domain_classification': 'categorical_crossentropy'},
              metrics={'classification': 'accuracy', 'domain_classification': 'accuracy'})

# 训练模型
model.fit(x_combined, [y_combined_class, domain_labels], epochs=10, batch_size=64)

# 预测结果
preds = model.predict(x_test_mnist[:10])
plot_results(x_test_mnist[:10], y_test_mnist[:10], domain_labels[:10], preds)


# In[ ]:




