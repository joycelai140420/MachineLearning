#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
使用 PyTorch 框架来演示如何在一个简单的循环神经网络（RNN）中实施梯度裁剪，
并可视化训练过程中的损失函数变化。我们将使用PyTorch自带的数据集进行示例，
例如使用MNIST数据集进行数字识别的任务，但为了演示梯度裁剪在序列数据上的应用，
我们将这个图像识别任务视作一个序列处理问题，每行像素作为一个时间步。
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 定义模型
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(28, 128, batch_first=True)  # 输入维度28（每行像素），隐层维度128
        self.fc = nn.Linear(128, 10)  # 输出层，10个数字类别

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # 只取序列的最后一个输出
        return x

model = SimpleRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型并应用梯度裁剪
def train(model, train_loader, criterion, optimizer, epochs=10):
    '''
    确保模型处于训练模式。这对于某些模块（如 Dropout 和 BatchNorm）是必要的，
    因为它们在训练和评估模式下的行为不同。
    '''
    model.train()
    loss_history = [] # 初始化一个列表，用来记录每个训练步骤的损失值。
    for epoch in range(epochs): # 迭代多个训练周期
        #从数据加载器中逐批次提取图像和标签，用于训练。
        for images, labels in train_loader:# 从数据加载器中逐批次获取图像和标签，用于训练。
            optimizer.zero_grad()#在新的训练批次开始前重置梯度，因为默认情况下梯度是累加的。
            '''
            将图像数据重塑成适合模型的形状并进行前向传播。
            这里假设使用的是一个每次接受28x28图像的简单RNN。
            '''
            outputs = model(images.view(-1, 28, 28))  
            loss = criterion(outputs, labels)#使用定义的损失函数计算预测输出和实际标签之间的损失。
            loss.backward()#执行反向传播，PyTorch自动计算所有可训练参数的梯度。
            '''
            梯度裁剪，限制梯度的最大范数为2.0，防止梯度在反向传播中变得过大。
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  
            optimizer.step()#根据计算的梯度更新网络参数。
            loss_history.append(loss.item()) #将当前损失值添加到损失历史列表中。
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return loss_history

loss_history = train(model, train_loader, criterion, optimizer)

# 可视化损失变化
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.title('Training Loss Progress')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




