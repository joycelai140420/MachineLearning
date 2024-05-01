#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''
这case 是为了说明其原理和步骤，我将提供一个简化的实现版本。我们将使用模拟数据进行此示例
'''
import random
import matplotlib.pyplot as plt

# 创建模拟数据 - 二维平面上的点
data = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(10)]

# 参数(必须要从业务或是需求层面，事先知道)
n_neighbors = 2  # 每个点的局部邻居数量
n_dimensions = 1  # 目标低维空间的维度

# 计算欧氏距离
def euclidean_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

# 找到每个点的最近邻居
def find_neighbors(point, data, n_neighbors):
    '''
    为数据集中的每个点（除了点本身）计算到point的欧几里得距离。
    euclidean_distance(point, other)是一个函数调用，它计算两点之间的距离。
    结果是一个元组列表，每个元组包含一个距离和对应的点（other）。
    列表中不包括点本身，避免自己到自己的距离计算。
    '''
    distances = [(euclidean_distance(point, other), other) for other in data if other != point]
    '''
    将上述创建的元组列表按照距离进行排序。
    默认情况下，sort()方法按元组的第一个元素（这里是距离）升序排序。
    '''
    distances.sort()
    '''
    经过排序后，这行代码使用列表推导式从排序后的列表中选择前n_neighbors个最近的点。
    distances[:n_neighbors]取出列表中的前n_neighbors个元组（即最近的几个邻居）。
    对于每个元组，_忽略了距离值（因为这里只需要点的信息），neighbor是元组中的点。
    返回值是一个包含每个点最近的n_neighbors个邻居的列表。
    '''
    return [neighbor for _, neighbor in distances[:n_neighbors]]

# 构建权重矩阵，假设每个邻居的贡献相等
#该函数接受 data（数据点列表）和 n_neighbors（每个点考虑的邻居数量）作为参数。
def construct_weights(data, n_neighbors):
    weights = {}#初始化权重
    for point in data: #对于数据集中的每个点 point，执行以下操作：
        #调用 find_neighbors 函数找到该点的 n_neighbors 个最近邻居。
        neighbors = find_neighbors(point, data, n_neighbors)
        #为每个邻居分配相等的权重，即每个邻居的权重为 1 / n_neighbors。
        #这里假设邻域内的每个点对重建当前点具有相同的贡献。
        weights[tuple(point)] = {tuple(neighbor): 1 / n_neighbors for neighbor in neighbors}
    return weights

# LLE优化：简化的版本，我们这里用随机生成的新坐标作为低维表示
# 真正的实现需要使用特征值分解来优化这个嵌入
def lle(data, n_neighbors, n_dimensions):
    #调用 construct_weights 函数为数据集中的每个点构建权重。
    weights = construct_weights(data, n_neighbors)
    #初始化低维嵌入：为数据集中的每个点随机生成一个低维空间中的坐标。
    #这里的坐标在 -1 到 1 之间随机生成，每个点生成 n_dimensions 维的坐标。
    low_dim_embedding = [[random.uniform(-1, 1) for _ in range(n_dimensions)] for _ in data]
    #返回低维嵌入：返回一个列表，列表中的每个元素是数据集中一个点在低维空间中的坐标。
    return low_dim_embedding


# In[9]:


def visualize(data, embedding):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Data')
    # 解包数据点列表为x和y坐标并绘图
    plt.scatter(*zip(*data))  # 正确展开数据列表为 x 和 y

    plt.subplot(122)
    plt.title('LLE Embedding')
    # 解包嵌入结果为x和y坐标并绘图
    if n_dimensions > 1:
        plt.scatter(*zip(*embedding))  # 如果嵌入维度大于1，正常展开
    else:
        plt.scatter(embedding, [0] * len(embedding))  # 如果嵌入维度为1，y坐标设为0

    plt.show()


# In[10]:


# 运行LLE算法
embedding = lle(data, n_neighbors, n_dimensions)
visualize(data, embedding)


# In[12]:


'''
使用 scikit-learn 库来实现 Locally Linear Embedding (LLE) 是一个非常直接的方法，因为 scikit-learn 提供了一个高效的 LLE 实现。
这种实现包括所有必要的数学优化，如特征值分解，可以直接应用于数据降维。
'''
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# 生成数据：Swiss Roll 数据集
n_samples = 1500
noise = 0.05
X, color = make_swiss_roll(n_samples, noise)

# 设置 LLE 参数
n_neighbors = 12  # 邻居的数量
n_components = 2  # 降维后的空间维度

# 初始化 LLE 对象
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')

# 执行 LLE 降维
X_transformed = lle.fit_transform(X)

# 可视化结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('Locally Linear Embedding')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(scatter)  # 确保 colorbar() 使用了scatter的返回值
plt.show()


# In[ ]:




