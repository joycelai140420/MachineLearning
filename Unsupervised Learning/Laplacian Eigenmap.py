#!/usr/bin/env python
# coding: utf-8

# In[4]:


#原始数据可视化
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图

# 数据生成
n_samples = 800
X, color = make_swiss_roll(n_samples, noise=0.1)

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 三维散点图
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
plt.title('Original Swiss Roll Data')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
plt.colorbar(scatter)  # 添加颜色条以显示点的高度

# 显示图形
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

# 数据生成，使用 make_swiss_roll 生成三维的 Swiss Roll 数据。
n_samples = 800
X, color = make_swiss_roll(n_samples, noise=0.1)

# 参数
n_neighbors = 10  # 邻近点数量
n_components = 2  # 目标维度

# 计算欧式距离矩阵
def compute_distances(X):
    dist = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
    return dist

# 构建邻接矩阵，基于每个点的 k 近邻创建邻接矩阵 W。
def adjacency_matrix(dist, k):
    n = dist.shape[0]
    neighbors = np.argsort(dist, axis=1)[:, 1:k+1]  # 找到k个最近邻
    W = np.zeros((n, n))
    for i in range(n):
        W[i, neighbors[i]] = 1
    W = (W + W.T) > 0  # 保证矩阵是对称的
    return W

# 构建拉普拉斯矩阵，拉普拉斯矩阵 L 由度矩阵 D 和邻接矩阵 W 计算得来。
def laplacian_matrix(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

# 执行拉普拉斯特征映射，通过对拉普拉斯矩阵进行特征分解，获取映射到低维空间的坐标。
def laplacian_eigenmap(L, dim):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvectors[:, 1:dim+1]

# 运算流程
dist = compute_distances(X)
W = adjacency_matrix(dist, n_neighbors)
L = laplacian_matrix(W)
Y = laplacian_eigenmap(L, n_components)

# 可视化原始数据
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title('Original Swiss Roll Data')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')

# 可视化降维结果
ax2 = fig.add_subplot(122)
ax2.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax2.set_title('Laplacian Eigenmaps Reduction')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')

plt.colorbar(ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral), ax=ax2)
plt.show()


# In[ ]:




