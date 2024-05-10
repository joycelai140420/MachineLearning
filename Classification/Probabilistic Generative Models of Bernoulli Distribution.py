#!/usr/bin/env python
# coding: utf-8

# In[6]:


#如何使用Python实现一个简单的伯努利概率生成模型，适用于处理具有二元特征的简单分类任务。
import numpy as np

class BernoulliGenerativeModel:
    #类的初始化方法，这里定义两个属性：
    #self.priors：用于存储每个类的先验概率。
    #self.likelihoods：用于存储每个类的特征的伯努利参数（即特征的条件概率）。
    def __init__(self):
        self.priors = None
        self.likelihoods = None
    #def fit(self, X, y):定义fit方法，用于训练模型。输入参数X是特征数据，y是对应的标签数据。
    def fit(self, X, y):
        #从输入数据X中获取样本数（n_samples）和特征数（n_features）。
        n_samples, n_features = X.shape
        #从标签数据y中提取唯一的类别，存储在self.classes中。
        #从标签数组 y 中提取出所有唯一的类别标签。y = [1, 2, 1, 3, 2, 3, 1],输出将会是 [1, 2, 3]
        self.classes = np.unique(y)
        #计算类别的数量。
        n_classes = len(self.classes)
        
        # 初始化先验和似然
        #初始化每个类别的先验概率为0。
        self.priors = np.zeros(n_classes)
        #初始化每个类别每个特征的伯努利参数为0。
        self.likelihoods = np.zeros((n_classes, n_features))

        # 计算先验和似然
        #遍历每个类别c及其索引idx。
        for idx, c in enumerate(self.classes):
            #从数据集X中提取属于类别c的样本。
            X_c = X[y == c]
            #计算类别c的先验概率，即该类别样本数除以总样本数。
            self.priors[idx] = X_c.shape[0] / n_samples
            #计算类别c中每个特征的平均值，用作该类别特征的伯努利参数。
            self.likelihoods[idx, :] = np.mean(X_c, axis=0)  # 伯努利参数估计为样本均值
    #定义predict方法，用于预测新数据X的类别。
    def predict(self, X):
        #获取预测数据中的样本数量。
        n_samples = X.shape[0]
        #初初始化后验概率矩阵，每行对应一个样本，每列对应一个类的后验概率。
        posteriors = np.zeros((n_samples, len(self.classes)))

        # 计算后验概率
        for idx, c in enumerate(self.classes):
            #prior: 类的先验概率的对数。
            prior = np.log(self.priors[idx])
            #likelihood: 计算给定特征值下，样本属于类 c 的对数似然概率。
            likelihood = np.sum(X * np.log(self.likelihoods[idx] + 1e-9) + 
                                (1 - X) * np.log(1 - self.likelihoods[idx] + 1e-9), axis=1)
            #更新 posteriors 矩阵，为每个样本加上该类的先验和似然。
            posteriors[:, idx] = prior + likelihood

        return self.classes[np.argmax(posteriors, axis=1)]


# In[7]:


# 示例数据
#设置随机种子以保证结果可复现。
np.random.seed(0)
#X_train: 生成一个100x10的二项分布随机矩阵，模拟伯努利分布的特征。
X_train = np.random.binomial(1, 0.5, (100, 10))
#y_train: 生成100个二项分布随机标签。
y_train = np.random.binomial(1, 0.5, 100)

# 模型训练和预测
model = BernoulliGenerativeModel()
#使用训练数据拟合模型。
model.fit(X_train, y_train)
#对训练数据进行预测。
predictions = model.predict(X_train)

# 准确率
accuracy = np.mean(predictions == y_train)
print(f'Accuracy: {accuracy}')


# In[ ]:




