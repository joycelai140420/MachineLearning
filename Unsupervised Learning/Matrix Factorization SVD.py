#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
假设我们有一个由用户对电影的评分构成的矩阵，其中大部分元素是空的。
应用矩阵分解后，我们可以得到两个矩阵：一个是用户矩阵（表示用户偏好），另一个是电影矩阵（表示电影属性）。
将两个矩阵相乘，我们可以为所有空白的元素（即用户未评分的电影）填充预测评分。
'''
import numpy as np
import matplotlib.pyplot as plt

# 假设数据：一个用户-电影评分矩阵，空白处用0表示未评分
# 用户对电影的评分（1-5分），0表示未评分
#后面要预测其空白的评分有可能是多少
np.random.seed(0)
original_scores = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])


# In[4]:


#使用 NumPy 的 np.linalg.svd 函数来分解评分矩阵。
#选择前k个奇异值：基于SVD结果，选择前k个最大的奇异值及其对应的奇异向量，用于降维和信息提取。

# 使用 SVD 分解矩阵
U, S, Vt = np.linalg.svd(original_scores, full_matrices=False)
S = np.diag(S)

# 选择前k个奇异值和奇异向量，这里我们选择前3个
k = 3
U_k = U[:, :k]
S_k = S[:k, :k]
Vt_k = Vt[:k, :]

# 通过 SVD 重构原始评分矩阵
'''
SVD 的分解元素
U：它是一个包含左奇异向量的矩阵。在我们的上下文中，每个左奇异向量代表用户的潜在特征空间。
Σ（S）：这是一个对角矩阵，其对角线上的元素是奇异值。奇异值衡量了每个潜在特征的重要性或“强度”。
在数据降维中，通常只保留最大的几个奇异值，因为它们包含了数据的主要信息。
VT ：它包含右奇异向量，每个向量代表物品的潜在特征空间。

具体步骤如下：
1.选择奇异值和向量：我们选择了前 k 个最大的奇异值及其对应的向量。
这意味着𝑈𝑘 、𝑆𝑘和 𝑉𝑡𝑘分别是从 𝑈、𝑆 和 𝑉𝑇中提取的前𝑘列（对于𝑈𝑘和 𝑉𝑡𝑘）和前 𝑘个奇异值
（对于 𝑆𝑘）。
2.点乘操作：
np.dot(S_k, Vt_k)：这一步首先计算了 Σ𝑘和 𝑉𝑘𝑇的乘积。由于 Σ𝑘是对角矩阵，
这一步实际上是对每个右奇异向量 𝑉𝑘𝑇中的元素进行了缩放，缩放因子是对应的奇异值。
np.dot(U_k, np.dot(S_k, Vt_k))：最后，将 𝑈𝑘与前一步的结果相乘。
这一步完成了从潜在特征空间到原始数据空间的映射，生成了重构的矩阵。

'''
predicted_scores = np.dot(U_k, np.dot(S_k, Vt_k))


# In[5]:


# 显示原始评分矩阵和预测评分矩阵
print("Original Ratings:")
print(original_scores)
print("Predicted Ratings:")
print(np.round(predicted_scores))

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_scores, cmap='viridis', interpolation='none')
plt.title("Original Ratings")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(predicted_scores, cmap='viridis', interpolation='none')
plt.title("Predicted Ratings")
plt.colorbar()


# In[ ]:




