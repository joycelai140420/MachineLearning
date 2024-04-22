#!/usr/bin/env python
# coding: utf-8

# In[1]:


#使用Python中的scikit-learn库来实现逻辑回归的简单示例
#call sklearn就不用详加解释
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 为了简化问题，我们只使用两个分类
X = X[y != 2]
y = y[y != 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
predictions = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')


# In[2]:


#用数学方法去实作逻辑回归的简单示例
import numpy as np
#定义了逻辑回归中使用的Sigmoid激活函数，它接收任意实数值z，将其映射到(0,1)区间，用于表示概率。
#np.exp(-z)计算-z的指数。
def sigmoid(z):
    """Sigmoid函数实现"""
    return 1 / (1 + np.exp(-z))

#compute_loss函数计算交叉熵损失，这是评估分类问题中模型表现的常用方法。
#y是真实标签的矩阵，y_hat是预测概率。
#m是样本的数量。
#交叉熵损失函数考虑了实际标签为1和为0的两种情况，
#其公式为：-Σ(y*log(y_hat) + (1-y)*log(1-y_hat)) / m。
#其交叉熵损失函数的直观解释是假设我们的模型预测了一个概率值，
#这个概率值表示某个事件发生的可能性。交叉熵损失函数将这个预测的概率和实际发生（或未发生）的事件的真实概率进行比较。
#如果预测的概率接近真实的概率，损失就小；如果预测的概率远离真实的概率，损失就大。
#y*log(y_hat)处理的是真实标签为1的情况跟预测的概率进行比较。
#(1-y)*log(1-y_hat)处理的是真实标签为0的情况跟预测的概率进行比较。
#相加就是这个预测的概率和实际发生（或未发生）的事件的真实概率进行比较。
#前面加上负号是因为对数函数log的值在（0,1)范围内是负的。添加负号是为了确保损失函数是正值。
def compute_loss(y, y_hat):
    """计算交叉熵损失"""
    m = y.shape[0]
    return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

#logistic_regression是主函数，用于训练逻辑回归模型。
#X是特征矩阵，y是标签矩阵。
#num_iterations是梯度下降的迭代次数，learning_rate是度下降的学习率。
#weights初始化为0，形状为(n, 1)，其中n是特征的数量。bias也初始化为0。
def logistic_regression(X, y, num_iterations, learning_rate):
    """逻辑回归模型训练"""
    m, n = X.shape
    # 初始化权重和偏差
    weights = np.zeros((n, 1))
    bias = 0
    #在每次迭代中，先计算线性预测值z。
    #使用sigmoid函数将z转换为概率y_hat。
    #调用compute_loss计算当前的损失。
    for i in range(num_iterations):
        # 线性组合
        #使用 np.dot() 在逻辑回归中执行矩阵乘法非常重要，因为它可以高效地并行处理所有样本的特征加权和。
        #这种矩阵操作比逐个计算每个样本的特征加权和要快得多，特别是在处理大型数据集时。
        z = np.dot(X, weights) + bias
        # 应用sigmoid函数
        y_hat = sigmoid(z)
        # 计算损失
        loss = compute_loss(y, y_hat)
        
        # 计算梯度并更新参数
        #dw和db分别是权重和偏差的梯度。
        #np.dot(X.T, (y_hat - y))计算权重梯度，np.sum(y_hat - y)计算偏差梯度。
        #使用梯度和学习率更新权重和偏差。
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        
        # 参数更新
        weights -= learning_rate * dw
        bias -= learning_rate * db
        #每100次迭代输出一次当前的损失，帮助监控训练过程。
        if i % 100 == 0:
            print(f"Loss after iteration {i}: {loss}")
    
    return weights, bias

# 创建模拟数据
#数据生成：我们生成一些随机数据，并定义一个简单的线性分类规则作为真实标签。
np.random.seed(1)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # 一个简单的线性决策规则

# 训练模型
num_iterations = 1000
learning_rate = 0.1
weights, bias = logistic_regression(X, y, num_iterations, learning_rate)

# 简单测试
index = 5
x_new = X[index, :]
y_pred = sigmoid(np.dot(x_new, weights) + bias)
print(f"Predicted: {y_pred}, True Label: {y[index]}")


# In[ ]:




