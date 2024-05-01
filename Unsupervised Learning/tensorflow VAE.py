#!/usr/bin/env python
# coding: utf-8

# In[9]:


'''
模型架构：这个VAE使用了一个简单的全连接网络作为编码器和解码器。编码器部分将图像从原始像素映射到潜在空间的均值和对数方差，
          而解码器则将潜在空间的样本映射回原始图像空间。
损失函数：损失函数包括重构损失（使用二进制交叉熵）和KL散度，两者结合用于训练网络。
训练和可视化：模型在 MNIST 数据上训练，并可视化重构的图像，以展示训练效果。
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy

# 参数
original_dim = 784
intermediate_dim = 64
latent_dim = 2
batch_size = 128
#因为跑有点久，设定为20，可以在设定大一点50
epochs = 20

# 加载 MNIST 数据
#数据加载和预处理：使用 TensorFlow 的 mnist.load_data() 函数加载 MNIST 数据集，将数据归一化并重塑为向量。
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 使用 TensorFlow Dataset API 处理数据
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)

# VAE 模型由 Encoder 和 Decoder 构成，包括编码器和解码器，采用全连接层，并通过一个采样函数实现潜在空间的采样。
# Encoder
'''AE的输入层。
Input 函数用于指定输入数据的形状 (shape)。这里，original_dim 是输入向量的维度，对于MNIST数据集来说，通常是 784（即 28x28 图像展平后的尺寸）。
'''
inputs = Input(shape=(original_dim,), name='encoder_input')
'''
定义一个全连接层（Dense层），
它的输出维度为 intermediate_dim（这是一个超参数，代表中间层的维度）。激活函数使用了ReLU，这是为了引入非线性，帮助模型学习更复杂的数据表示。
'''
x = Dense(intermediate_dim, activation='relu')(inputs)
'''
这两行定义了两个全连接层，分别输出潜在空间的均值 (z_mean) 和对数方差 (z_log_var)。
这两个层的输出维度都是 latent_dim，它定义了潜在空间的维度。z_mean 用于生成潜在变量的均值，而 z_log_var 用于生成潜在变量的方差的对数。
'''
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# 用于采样的层
'''
这个 sampling 函数实现了重参数化技巧，这是VAE中的关键技术。它允许模型在训练时通过反向传播优化潜在变量。
epsilon 是从标准正态分布中采样的噪声，z_mean 和 exp(0.5 * z_log_var) 分别是潜在变量的均值和标准差。
'''
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
'''
使用 Lambda 层包装 sampling 函数，将 z_mean 和 z_log_var 作为输入，输出潜在层 z。Lambda 层允许你在模型中使用任意表达式作为层。
'''
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder
'''
解码器部分首先使用一个全连接层将潜在空间的输出转换回中间维度（intermediate_dim），
然后通过另一个全连接层将其转换回原始维度（original_dim），使用sigmoid激活函数确保输出值在 [0, 1] 之间，因为输入数据已被归一化到这个范围。
'''
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_output = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded = decoder_output(h_decoded)

# VAE 模型，定义整个 VAE 模型，将输入层与解码输出层连接。
vae = Model(inputs, x_decoded, name='vae_mlp')

# VAE 损失函数，结合重构损失和 KL 散度，直接添加到模型上。
'''
定义损失函数，包括重构损失和KL散度。重构损失使用二进制交叉熵计算输入和输出之间的差异。KL散度计算潜在分布与先验分布（通常是标准正态分布）之间的差异。
两者结合，形成了VAE的总损失。然后使用 add_loss() 将总损失添加到模型上，并编译模型，使用 adam 优化器。
'''
#计算重构损失，这里使用二元交叉熵损失函数，并将其缩放到原始输入维度。
reconstruction_loss = binary_crossentropy(inputs, x_decoded)
reconstruction_loss *= original_dim
#计算 KL 散度，它衡量潜在分布与标准正态分布之间的差异。
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_sum(kl_loss, axis=-1)
kl_loss *= -0.5
#总损失为重构损失和 KL 散度的和，通过 add_loss 添加到模型上，并编译模型，使用 'adam' 优化器
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
#输出模型的概览信息。
vae.summary()

# 训练 VAE，通过 .fit() 方法，直接使用 TensorFlow 的数据集进行训练和验证。

vae.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

# 可视化函数，训练完成后，展示原始和由 VAE 重构的图像。
def plot_images(x_test, decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 显示原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.axis('off')

        # 显示重构后的图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.axis('off')
    plt.show()

# 使用 VAE 生成图像
decoded_images = vae.predict(x_test)
plot_images(x_test, decoded_images)


# In[ ]:




