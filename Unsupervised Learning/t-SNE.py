#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图

# 数据生成
n_samples = 800
X, color = make_swiss_roll(n_samples, noise=0.1)

# 实例化并运行 t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X)

# 可视化原始数据
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title('Original Swiss Roll Data')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')

# 可视化 t-SNE 降维结果
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap=plt.cm.Spectral)
ax2.set_title('t-SNE Reduction')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
plt.colorbar(scatter)
plt.show()


# In[ ]:




