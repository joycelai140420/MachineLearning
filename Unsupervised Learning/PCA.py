#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 加载数据
#加载 Iris 数据集并对特征进行标准化，确保 PCA 不受不同尺度的影响。
iris = load_iris()
X = iris.data
y = iris.target

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 实施 PCA
#使用 scikit-learn 的 PCA 类降低数据到两个主成分。
pca = PCA(n_components=2)  # 降到2维
X_pca = pca.fit_transform(X_scaled)

# 可视化降维结果,绘制降维后的数据，颜色按类别区分。
'''
使用的是 Iris 数据集，这个数据集包含三种不同的鸢尾花种类。在绘制 PCA 结果的散点图时，我使用不同的颜色来区分这三种鸢尾花种类。
这三种类别在 PCA 降维后的二维空间中以不同的颜色显示，使我们可以直观地看到不同类别数据在低维空间的分布情况。
使用 Matplotlib 的 scatter 函数并通过 c=y 参数指定颜色，其中 y 是类别标签数组，cmap='viridis' 指定了颜色映射，这使得每个类别都有一个独特的颜色。
通过图例或颜色条可以清楚地识别每种颜色对应的鸢尾花种类。
'''
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=150)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)
plt.show()

# 计算重构误差，通过将数据投影回原始维度并计算均方误差，评估降维质量。
X_projected = pca.inverse_transform(X_pca)
reconstruction_error = mean_squared_error(X_scaled, X_projected)
print(f"Reconstruction Error (Loss): {reconstruction_error:.4f}")

# 解释方差比例,显示每个主成分解释的方差比例。
print("Explained variance ratio:", pca.explained_variance_ratio_)


# In[2]:


#use SVD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 创建模拟数据
#使用 make_blobs 生成一组具有4个中心的高维数据。
#虽然这里没有使用真实数据集，但这个模拟的数据集足以展示 PCA 的工作原理。
X, _ = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=2.0)

# 数据中心化（PCA的重要前提）
'''
PCA 需要数据中心化处理，即每个特征的均值需要减去其平均值。
这是因为 PCA 要在数据的协方差矩阵上工作，而协方差矩阵是基于中心化数据计算的。
'''
X_centered = X - np.mean(X, axis=0)

# 使用 NumPy 的奇异值分解（SVD）函数
'''
np.linalg.svd 用于计算中心化数据的奇异值分解。
U 是左奇异向量，S 是奇异值，
Vt 是右奇异向量的转置（即主成分方向）。
'''
U, S, Vt = np.linalg.svd(X_centered)

# 选择前两个主成分
'''
Vt.T 的前两列包含了最重要的两个主成分（最大的奇异值对应的）。
我们选择这两个主成分来进行降维。
'''
W2 = Vt.T[:, :2]

# 将数据投影到选定的主成分上
'''
通过将中心化的数据点与主成分矩阵 W2 相乘，实现将数据投影到新的低维空间。
'''
X_pca = X_centered.dot(W2)

# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', alpha=0.5)
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()


# In[ ]:




