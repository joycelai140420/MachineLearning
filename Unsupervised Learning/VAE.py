#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
提供一个基础概念的解释和简化的前向传播代码，这样的实现不会具有实际的训练功能。
为了完整地训练 VAE，你通常需要使用像 TensorFlow 或 PyTorch 这样的框架。
构建一个非常基础的 VAE 结构，并使用 MNIST 数据集进行可视化。
请注意，这个示例主要用于解释 VAE 的结构和前向传播过程，不包含反向传播或任何学习步骤。
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 数据加载
#使用 TensorFlow 的 mnist.load_data() 函数加载 MNIST 数据集
def load_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test

x_train, x_test = load_data()

# VAE 模型参数
# 是模型输入层的维度，它应该与数据集中单个样本的特征数量相匹配。
#在 MNIST 数据集中，每个图像由28x28个像素点组成，总共784个像素值。
#每个像素值被平展（flattened）成一个一维数组，因此 input_dim 设置为784。
input_dim = x_train.shape[1]  # 784 for MNIST
#指的是隐藏层的维度，即在输入层和潜在层之间的神经网络层的大小。这里设置为64，意味着数据从784维被压缩到64维。
hidden_dim = 64
#也就是模型在编码过程中数据被压缩到的最终内部表示的维度。设置为2意味着模型试图将输入数据压缩到一个二维潜在空间中。
latent_dim = 2

# 权重初始化
#分别是输入维度 (input_dim)、隐藏层维度 (hidden_dim) 和潜在空间维度 (latent_dim)。
'''
权重组件
'encoder_h'：尺寸为 (input_dim, hidden_dim) 的矩阵。用于编码器部分，将输入数据从原始输入空间映射到隐藏层空间。
'encoder_mean'：尺寸为 (hidden_dim, latent_dim) 的矩阵。用于从编码器的隐藏层计算潜在空间的均值，这个均值定义了潜在空间中的正态分布的中心。
'encoder_log_var'：尺寸为 (hidden_dim, latent_dim) 的矩阵。用于从编码器的隐藏层计算潜在空间的对数方差，这个对数方差定义了潜在空间中的正态分布的分散程度。
'decoder_h'：尺寸为 (latent_dim, hidden_dim) 的矩阵。用于解码器部分，负责将潜在空间的表示映射回较高的隐藏层维度。
'decoder_out'：尺寸为 (hidden_dim, input_dim) 的矩阵。用于解码器的最后一层，将隐藏层的表示映射回原始数据空间，尝试重构输入数据。
'''
def init_weights(input_dim, hidden_dim, latent_dim):
    weights = {
        'encoder_h': np.random.normal(size=(input_dim, hidden_dim)),
        'encoder_mean': np.random.normal(size=(hidden_dim, latent_dim)),
        'encoder_log_var': np.random.normal(size=(hidden_dim, latent_dim)),
        'decoder_h': np.random.normal(size=(latent_dim, hidden_dim)),
        'decoder_out': np.random.normal(size=(hidden_dim, input_dim))
    }
    return weights
'''
使用 np.random.normal 来初始化权重，这是一个常用的初始化方法，它从标准正态分布中随机抽取权重值。
这样的初始化有助于避免训练初期的梯度消失或爆炸问题，因为正态分布产生的权重既有正值也有负值，均值为 0，有助于保持激活值分布的均衡。
'''
weights = init_weights(input_dim, hidden_dim, latent_dim)

# sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# encoder
def encoder(x):
    '''
    x：输入数据，通常是一个批次的数据，每行代表一个样本，每列代表一个特征。
    隐藏层转换：np.dot(x, weights['encoder_h'])：这一步是一个典型的神经网络前向传播步骤，
    它将输入数据 x 与编码器的权重 weights['encoder_h'] 相乘。
    这里的 weights['encoder_h'] 是从输入层到隐藏层的权重矩阵。
    sigmoid(...)：激活函数被应用于上一步的输出。
    在这里使用 Sigmoid 函数作为激活函数是为了将隐藏层的输出压缩到 (0, 1) 的范围内，帮助网络学习非线性特征。
    
    '''
    h = sigmoid(np.dot(x, weights['encoder_h']))
    '''
    均值和对数方差计算：mean = np.dot(h, weights['encoder_mean']) 和 log_var = np.dot(h, weights['encoder_log_var'])：
    这两步进一步处理激活函数的输出 h，分别计算潜在空间中的均值和对数方差。
    weights['encoder_mean'] 和 weights['encoder_log_var'] 是从隐藏层到潜在空间均值和对数方差的权重矩阵。
    这些参数（均值和对数方差）随后用于生成潜在空间中的样本，这些样本将被用来重构输入数据，同时优化的目标是最小化重构误差和潜在表示的 KL 散度。
    '''
    mean = np.dot(h, weights['encoder_mean'])
    log_var = np.dot(h, weights['encoder_log_var'])
    return mean, log_var

# 重参数化技巧
'''
在VAE中，编码器输出的是潜在变量的参数，通常是一个均值（mean）和一个对数方差（log_var）。
理论上，接下来应该从这个参数化的分布中抽取样本，但直接从分布中抽取样本会断开计算图，使得梯度无法直接反向传播。
重参数化技巧就是为了解决这个问题，重参数化技巧将抽样过程分解为两步：
1.确定性的变换：首先生成一个不依赖于任何模型参数的噪声变量 𝜖。这个噪声通常来自一个标准的分布，比如标准正态分布。
2.随机性的引入：然后，通过一个参数化的变换将噪声变量转换成实际的样本。具体到VAE，这个变换由以下公式给出：
z=μ+σ⋅ϵ
其中，𝜇 是均值，𝜎 是标准差，它是从对数方差log(𝜎2)log(σ2 ) 转换而来的，即𝜎=exp(log(𝜎2)/2)。
这样做的结果是，VAE可以在保持抽样操作的随机性的同时，有效地通过反向传播算法学习到如何产生类似于训练数据的新数据点。
'''
def reparameterize(mean, log_var):
    eps = np.random.normal(size=mean.shape)
    return mean + np.exp(log_var / 2) * eps

# decoder
'''
z: 这是从潜在空间中抽样得到的点。在 VAE 中，这些点通常通过编码器输出的均值和方差进行重参数化技巧后获得。
隐藏层 h：
h = sigmoid(np.dot(z, weights['decoder_h']))
这行代码首先通过矩阵乘法 np.dot(z, weights['decoder_h']) 将潜在空间的点 z 与解码器的第一层权重 weights['decoder_h'] 相乘。
这一步是为了将潜在变量转换回更高维的表征空间。使用 sigmoid 激活函数处理上述矩阵乘法的结果。
Sigmoid 函数将输出压缩到 (0, 1) 的范围内，有助于处理那些需要输出为概率或者像像素强度这样必须位于特定范围的应用。
重构 reconstruction：
reconstruction = sigmoid(np.dot(h, weights['decoder_out']))
这行代码再次应用矩阵乘法，这次是将隐藏层 h 与解码器的第二层权重 weights['decoder_out'] 相乘。
此步骤旨在将数据从隐藏层的高维表征进一步转换回原始数据的维度。
再次应用 sigmoid 激活函数，确保最终输出的每个元素都位于 (0, 1) 之间，这对于处理归一化的图像数据尤其重要。
返回 reconstruction：
函数返回 reconstruction，这是从潜在变量 z 重构回的数据。
在处理图像的情况下，这些重构的数据可以直接与原始图像进行比较，以计算如二元交叉熵之类的损失，从而在训练过程中更新权重。

'''
def decoder(z):
    h = sigmoid(np.dot(z, weights['decoder_h']))
    reconstruction = sigmoid(np.dot(h, weights['decoder_out']))
    return reconstruction
'''
这个 plot_latent_space 函数的目的是在二维潜在空间中可视化由变分自编码器（VAE）生成的图像。
通过在潜在空间的网格上均匀采样，函数解码每个采样点对应的图像并展示在一个大的图像网格中。
这种可视化帮助我们理解潜在空间如何编码图像的不同特征
'''
# 可视化
def plot_latent_space(x_test, n=15, figsize=10):
    # 显示 n x n 个数字
    #设定每个数字的大小为 28x28 像素（因为 MNIST 数据集中的图像为 28x28），然后创建一个足够大的零矩阵 figure 来容纳整个 n x n 的图像网格。
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # 线性间距的坐标用于构造一个 n x n 图像的网格
    #使用 np.linspace 在 [-3, 3] 区间内生成 n 个均匀间隔的点。
    #这些点代表潜在空间中的坐标，这个区间选择是基于潜在空间的分布（通常假设为标准正态分布）。
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    #使用 decoder 函数将这个潜在点解码成一个图像。decoder 应该输出与 MNIST 数据集图像相同尺寸的重构图像。  
    #将重构的图像 x_decoded 调整形状至 28x28，并将其放置在 figure 矩阵的相应位置。
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.show()

plot_latent_space(x_test)


# In[2]:





# In[ ]:




