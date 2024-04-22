#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# 示例数据集
data = np.array([
    [50, 30],
    [20, 90],
    [30, 80],
    [40, 70],
    [60, 60],
])


# In[2]:


#Z-Score Normalization：数据的每个特征值减去其均值，然后除以其标准差。
#通过计算数据的平均值和标准差，并使用这些值来转换每个特征，使得每个特征的平均值为0且标准差为1。
def z_score_normalization(data):
    """ 对数据进行Z-score标准化 """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds


# In[3]:


#Min-Max Scaling：将特征缩放到给定的最小值和最大值之间，通常是0和1。
#计算每个特征的最小值和最大值，并使用这些值将特征缩放到0和1之间（或任何其他指定的范围）
def min_max_scaling(data, feature_range=(0, 1)):
    """ 对数据进行Min-Max标准化 """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scale = feature_range[1] - feature_range[0]
    return (data - min_val) / (max_val - min_val) * scale + feature_range[0]


# In[4]:


# 应用Z-score标准化
z_normalized_data = z_score_normalization(data)
print("Z-score Normalized Data:")
print(z_normalized_data)

# 应用Min-Max标准化
min_max_normalized_data = min_max_scaling(data, feature_range=(0, 1))
print("Min-Max Normalized Data:")
print(min_max_normalized_data)


# In[ ]:




