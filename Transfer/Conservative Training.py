#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
构建一个简单的卷积神经网络（CNN）来进行迁移学习示例
加载 VGG16 预训练模型，对其顶层进行修改，并进行Fine-tuning。

'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 加载并预处理数据
'''
使用 TensorFlow/Keras 的 cifar10 模块自动下载并加载 CIFAR-10 数据集。
CIFAR-10 是一个广泛使用的数据集，包含60000张32x32的彩色图像，分为10个类别，
每类6000张图像。这里，数据被分为训练集 (x_train, y_train) 和测试集 (x_test, y_test)。
将图像数据从整数（0到255的像素值）转换为浮点数，并通过除以255来归一化到[0,1]区间。
深度学习常用的预处理步骤，有助于模型训练的稳定性和性能。
to_categorical 函数将类别向量（整数）转换为二进制类矩阵，这对于需要分类输出的模型来说是必需的。这里的10表示有10个类别。
例如，如果一个样本属于第3类，它的类别向量将被转换为[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]。
'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 加载VGG16模型，不包括顶层
'''
VGG16 是一个在 ImageNet 数据集上预训练的深度卷积神经网络，广泛用于图像识别任务。
这里，weights='imagenet' 指定加载模型时使用在 ImageNet 数据集上预训练的权重。
include_top=False 指明在加载模型时不包括顶部（即，不包括分类层，去除最顶部的全连接层）的网络层，
这允许我们添加自己的分类层以适应 CIFAR-10 数据集的10个类别。
也可以透过以下方式选择性使用层，通过访问模型的 .layers 属性来单独使用或重新组合这些层。
例如，提取前三层的输出：
# 加载完整的预训练模型
base_model = VGG16(weights='imagenet', include_top=True)  # 这次包括顶部层
# 提取前三层
model_input = base_model.input
layer_output = base_model.layers[2].output  # 第三层的输出
# 创建一个新模型，只包括前三层
new_model = Model(inputs=model_input, outputs=layer_output)

如果你想调整某些层而冻结其他层，可以这样做：
# 冻结除前三层外的所有层
for layer in base_model.layers[3:]:
    layer.trainable = False
如果你想从模型的某一层开始构建新模型，可以这样设置：
# 从第三层开始构建新模型
third_layer_output = base_model.layers[2].output
x = Dense(1024, activation='relu')(third_layer_output)
predictions = Dense(10, activation='softmax')(x)

# 创建新模型
new_model_from_third_layer = Model(inputs=base_model.input, outputs=predictions)

input_shape=(32, 32, 3) 明确设置了输入数据的形状，即32x32像素的RGB图像。这是必要的，
因为VGG16默认的输入尺寸是224x224，我们需要调整它以适应CIFAR-10的图像尺寸。
'''
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 冻结基础模型的所有层
#在预训练的 VGG16 网络上添加新的全连接层和输出层，并冻结原始的 VGG16 层以保留其学习到的特征。
#这意味着在后续的训练过程中，这些层的权重不会被更新，即这些层被“冻结”。
#只有那些新添加的层（比如全连接层和输出层）的权重会在训练过程中更新，以适应新任务。
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = base_model.output
'''
GlobalAveragePooling2D(): 这是一个池化层，它对每个特征图的空间维度（宽度和高度）进行平均，
从而将特征图转换成单个数字。如果输入的特征图维度是 HxWxC（高度、宽度、通道数），
Global Average Pooling 的输出将是 1x1xC。
这样做的好处是显著减少模型的参数量，并且有助于减少过拟合，同时保留重要的特征。
'''
x = GlobalAveragePooling2D()(x)
'''
Dense(1024, activation='relu'): 这是一个全连接层，它有1024个神经元。
全连接层是神经网络中常见的层，用于学习输入数据的非线性组合。
这里使用ReLU（Rectified Linear Unit）激活函数，在前面线性那个单元有说明，
ReLU 函数提供了简单的非线性，允许模型学习复杂的数据模式。
'''
x = Dense(1024, activation='relu')(x)  # 新增全连接层
'''
Dense(10, activation='softmax'): 这是模型的输出层。
它同样是一个全连接层，但神经元数量设置为10，对应于CIFAR-10数据集的10个类别。
使用 softmax 激活函数，这是多分类任务中常用的激活函数。
softmax 函数能够将输出值转化为概率分布，
每个输出值代表了模型预测输入图像属于每个类别的概率。
'''
predictions = Dense(10, activation='softmax')(x)

# 构建最终模型
'''
这行代码定义了整个模型的输入和输出。base_model.input 指的是预训练模型（如 VGG16）的输入层，
这保证了我们的新模型可以接受与预训练模型相同形式的输入数据。
predictions 是我们添加在预训练模型后面的全连接网络层的输出。
这个输出层设计为对应新任务的类别数量，这里是 CIFAR-10 的 10 类分类。
通过将输入和输出连接起来，我们实现了从原始输入到最终分类预测的直接映射，
完整地定义了一个新的深度学习模型。
'''
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
'''
model.compile 是准备模型进行训练的步骤。在这里，你需要指定至少三个参数：优化器、损失函数和评价指标。

optimizer=Adam(lr=0.0001) 定义了优化器为 Adam，它是一种广泛使用的更新权重的方法，
前面课程有说明，
帮助模型在训练过程中更快地收敛。lr=0.0001 设置了学习率为 0.0001，这是一个相对较小的值，
有助于模型在从预训练网络迁移学习时进行细微调整，避免对已经学习好的特征造成破坏。

loss='categorical_crossentropy' 设置损失函数为分类交叉熵，这是多类分类问题中常用的损失函数，
用于评估模型预测的概率分布与实际标签的概率分布之间的差异。

metrics=['accuracy'] 表示在训练和测试过程中，模型会计算并报告准确率，即正确分类的样本比例。
'''
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
'''
model.fit 是用来训练模型的函数。这里，x_train 和 y_train 是训练数据及其标签，模型将使用这些数据来学习。
epochs=10 指定训练过程将遍历整个数据集十次。每次遍历称为一个 epoch，多个 epoch 可以帮助模型更好地学习数据。
batch_size=32 设置每个批次包含32个数据点。批次大小是深度学习中的一个重要参数，影响模型的训练速度和内存需求。
前面都有教过，基本上运算量就是10*32
validation_data=(x_test, y_test) 指定了用于评估模型性能的测试数据集。在每个 epoch 结束后，
模型会在这个数据集上计算损失和其他指标，帮助监控模型在未见数据上的表现，防止过拟合。

'''
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 可视化输入和训练后的结果
def plot_images(images, labels, preds):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Actual: {np.argmax(labels[i])}, Predicted: {np.argmax(preds[i])}")
        plt.axis('off')
    plt.show()

preds = model.predict(x_test[:9])
plot_images(x_test[:9], y_test[:9], preds)


# In[ ]:




