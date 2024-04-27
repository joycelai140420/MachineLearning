#!/usr/bin/env python
# coding: utf-8

# In[5]:


#use toolkit implement Fully Connected Feedforward Network
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[6]:


# 定义网络结构
#nn.Module：所有神经网络模块的基类，网络模型应继承自这个类。
class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self):
        #super()：调用基类的初始化函数，这在Python类的继承中是常见做法。
        super(FeedforwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 定义第一个全连接层，从输入层到隐藏层，输入特征为10，输出特征为50。
        self.relu = nn.ReLU()         # ReLU激活函数
        self.fc2 = nn.Linear(50, 1)   # 定义第二个全连接层，从隐藏层到输出层，输出特征为1（适合二分类问题这里用Linear）。
        self.sigmoid = nn.Sigmoid()   # Sigmoid激活函数
    #定义前向传播过程（这就是README.md中说的手动连接）
    def forward(self, x):
        out = self.fc1(x) #数据首先通过第一个全连接层。
        out = self.relu(out)#然后通过ReLU激活函数。
        out = self.fc2(out) #经过ReLU处理后的数据传入第二个全连接层。
        out = self.sigmoid(out) #最后通过Sigmoid激活函数，输出预测结果。
        return out


# In[7]:


# 实例化模型。
model = FeedforwardNeuralNetwork()

# 损失函数和优化器
criterion = nn.BCELoss()  # 定义二分类问题的交叉熵损失函数。
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD梯度下降优化器，指定学习率为0.01，并将模型的参数传入优化器。

# 示例数据
x = torch.randn(100,10)  #  随机生成输入数据，模拟100个数据点，每个有10个特征，就是生成一个包含10个特征的随机数据点。
#y = torch.tensor([1.0])  # 目标数据，表示这是正类
y = torch.rand(100, 1).round()  # 100个随机二分类标签，生成目标标签，这里使用1.0表示正类。

# 存储损失以便可视化
losses = []


## 前向传播（单数据点）
#outputs = model(x) #通过模型传递输入数据 x。
#loss = criterion(outputs, y)#计算预测输出 outputs 和真实标签 y 之间的损失。
# 反向传播和优化
#optimizer.zero_grad()  # 清空过往梯度，否则梯度会在每次.backward()调用时累加。
#loss.backward()  # 执行反向传播，计算每个参数的梯度。
#optimizer.step()  # 根据计算得到的梯度更新参数。

# 训练网络
for epoch in range(500):  # 训练500轮
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录损失
    losses.append(loss.item())

    # 每50轮打印一次损失
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.plot(losses)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[12]:


#not use toolkit implement Fully Connected Feedforward Network
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


def sigmoid(x):
    """Sigmoid激活函数
    将输入压缩到0和1之间，常用于二分类任务的输出层。
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid激活函数的导数
    用于反向传播时计算梯度。
    """
    return x * (1 - x)

def relu(x):
    """ReLU激活函数
    非线性激活函数，如果输入小于0则输出0，否则输出输入值。常用于隐藏层。
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU激活函数的导数
    在反向传播中使用，输出是0或1，这取决于输入是否大于0。
    """
    return (x > 0).astype(float)


# In[14]:


class FullyConnectedFeedforwardNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        """
        初始化网络参数，包括权重和偏置。权重通过正态分布随机初始化并乘以0.1
        （缩小初始权重），偏置初始化为0。
        input_size, hidden_size, output_size：分别代表输入层、隐藏层和输出层的神经元数量。
        """
        """
        生成一个形状为 (input_size, hidden_size) 的数组，数组中的元素是从标准正态分布（均值为0，标准差为1）中抽取的。
        input_size 是输入层的神经元数量，hidden_size 是隐藏层的神经元数量。
         * 0.1： 将权重的初始值缩小，乘以0.1是为了避免初始权重过大，这样可以帮助模型在训练初期更稳定地学习。
        较小的权重起始值有助于保持神经网络输出的非线性激活函数在其线性区域内，从而保证梯度稳定。
        """
        self.weights_input_to_hidden = np.random.randn(input_size, hidden_size) * 0.1
        """
        类似于第一行，这里也是创建一个从隐藏层到输出层的权重矩阵。hidden_size 表示隐藏层神经元数量，output_size 表示输出层神经元数量，
        通常在二分类问题中为1。
        权重再次乘以0.1是为了控制初始权重的大小，以助于梯度稳定和模型训练的收敛。
        """
        self.weights_hidden_to_output = np.random.randn(hidden_size, output_size) * 0.1
        """
        np.zeros((1, hidden_size))：创建一个形状为 (1, hidden_size) 的数组，其中所有元素都为0。
        这意味着隐藏层的每个神经元初始时都有一个为0的偏置。
        偏置的作用是在激活函数应用之前向神经元的输入添加一个常数偏移。初始化为零是常见的做法，因为激活函数（如ReLU）在0附近通常是对称的。
        """
        self.bias_hidden = np.zeros((1, hidden_size))
        """
        类似于隐藏层偏置的初始化，这里为输出层的每个神经元设置了初始偏置为0。
        在训练过程中，这些偏置值将根据数据和损失函数通过反向传播进行调整。
        """
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        """前向传播
        输入：接受输入数据 x。参数 x 是输入数据，通常是一个二维数组（矩阵），其中每一行代表一个样本，每一列代表一个特征。
        隐藏层处理：计算输入到隐藏层的线性变换，然后应用ReLU激活函数。
        输出层处理：计算隐藏层到输出层的线性变换，然后应用Sigmoid激活函数以产生最终的预测输出。
        """
        self.input = x
        """
        这行执行输入层到隐藏层的线性变换。
        使用 np.dot(x, self.weights_input_to_hidden) 计算输入数据 x 和权重矩阵 self.weights_input_to_hidden 的矩阵乘法，
        这会为每个输入样本生成隐藏层的线性响应。
        + self.bias_hidden 添加偏置项到每个隐藏层神经元的线性响应上。
        由于广播（broadcasting），偏置从一个形状与隐藏层神经元数相同的向量扩展到每个样本。
        """
        self.hidden_layer_input = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
        """
        应用ReLU激活函数到隐藏层的线性输出上。ReLU（Rectified Linear Unit）函数是一个非线性函数，定义为 max(0, x)，
        用于增加模型的非线性能力，允许网络学习更复杂的函数。
        relu(self.hidden_layer_input) 计算每个隐藏层神经元的激活输出，负值被置为0，非负值保持不变。
        """
        self.hidden_layer_output = relu(self.hidden_layer_input)
        """
        类似于前一个线性变换，这行代码执行从隐藏层到输出层的线性变换。只是把上面的output接过来。
        """
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
        """
        应用Sigmoid激活函数到输出层的线性输出上。Sigmoid函数将每个输出压缩到0和1之间，常用于二分类任务中，输出可以解释为属于某类的概率。
        sigmoid(self.output_layer_input) 计算输出层每个神经元的激活输出，这是网络对给定输入的最终预测。
        """
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def backward(self, y):
        """反向传播
        误差计算：计算输出层和隐藏层的误差。
        梯度计算：使用激活函数的导数计算梯度。
        权重和偏置更新：根据梯度和学习率更新模型的权重和偏置。
        """
        """
        计算输出层误差
        output_error: 计算实际标签 y 和网络预测输出 self.output 之间的差值。这个误差是损失函数对输出的直接导数，即梯度的起点。
        """
        output_error = y - self.output
        """
        计算输出层的梯度
        output_delta: 这是输出层梯度，计算方法是输出误差 output_error 乘以输出层激活函数（这里是sigmoid）的导数。
        这个操作实现了链式法则，将误差转化为对应于每个输出神经元的梯度。
        """
        output_delta = output_error * sigmoid_derivative(self.output)

        """
        计算隐藏层误差
        hidden_error: 将输出层的梯度 output_delta 通过权重 self.weights_hidden_to_output 反向传播到隐藏层。
        这里使用了矩阵乘法，并转置权重矩阵，以正确地将错误传播到隐藏层的每个神经元。
        """
        hidden_error = output_delta.dot(self.weights_hidden_to_output.T)
        """
        计算隐藏层的梯度
        hidden_delta: 这是隐藏层梯度。计算方法是隐藏层的误差 hidden_error 乘以隐藏层激活函数（这里是ReLU）的导数。
        这一步同样应用了链式法则，将隐藏层的误差转换为每个隐藏神经元的梯度。
        """
        hidden_delta = hidden_error * relu_derivative(self.hidden_layer_output)

        # 更新权重和偏置
        """
        更新隐藏层到输出层的权重
        使用隐藏层的输出 self.hidden_layer_output（已经通过激活函数处理）和输出层梯度 output_delta 的乘积，更新隐藏层到输出层的权重。
        乘以学习率 learning_rate 控制更新的步长。
        
        """
        self.weights_hidden_to_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        """
        更新输出层偏置
        更新输出层的偏置，加上所有样本的输出层梯度之和，乘以学习率。np.sum(output_delta, axis=0, keepdims=True) 确保偏置维度与权重矩阵保持一致。
        """
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        """
        更新输入层到隐藏层的权重
        使用输入层的数据 self.input（已转置）和隐藏层的梯度 hidden_delta 的乘积，更新输入层到隐藏层的权重。这也乘以了学习率。
        """
        self.weights_input_to_hidden += self.input.T.dot(hidden_delta) * learning_rate
        """
        更新隐藏层偏置
        更新隐藏层的偏置，加上所有样本的隐藏层梯度之和，乘以学习率。
        """
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, learning_rate, epochs):
        """
        self: 表示类的实例本身，在类的方法中使用，用来访问类的属性和方法。
        x: 输入数据集，其中包含多个训练样本，每个样本包括多个特征。
        y: 标签数据集，包含每个训练样本对应的目标值或标签。
        learning_rate: 学习率，用于控制权重更新的步长。
        epochs: 训练的总轮数，即完整数据集用于训练模型的次数。
        """
        """
        losses: 用于存储每个训练周期的平均损失值的列表。这个列表在训练过程中会不断更新，训练结束后可以用来分析模型的训练效果或进行可视化。
        """
        losses = []
        """
        一个循环，从 0 到 epochs - 1。每次循环代表一个训练周期，即模型看到整个训练数据集一次。
        """
        for epoch in range(epochs):
            #前向传播计算输出
            output = self.forward(x)
            """
            loss = np.mean((y - output) ** 2): 计算损失函数的值。这里使用的是均方误差（MSE），它衡量的是模型输出和实际标签之间差的平方的平均值。
            这是回归任务中常用的损失函数，也可以用于二分类问题中输出层使用Sigmoid函数的场景。
            """
            loss = np.mean((y - output) ** 2)
            self.backward(y)#执行反向传播算法。
            losses.append(loss)
        return losses


# In[15]:


# 参数设置
input_size = 10
hidden_size = 5
output_size = 1
learning_rate = 0.01
epochs = 1000

# 数据生成
np.random.seed(0)
x = np.random.randn(100, input_size)
y = np.random.randint(0, 2, (100, output_size))

# 创建和训练网络
network = FullyConnectedFeedforwardNetwork(input_size, hidden_size, output_size)
losses = network.train(x, y, learning_rate, epochs)

# 绘制损失图
plt.plot(losses)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[ ]:




