#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[46]:


# 加载Iris数据集。
iris = datasets.load_iris()
X = iris.data[:, :2]  # 为了可视化方便，仅使用前两个特征
y = iris.target


# In[47]:


#这个函数将用于随后的可视化不同树深度的分类效果：
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()


# In[48]:


#使用Random Forest分类器进行训练，并设置不同的树深度，且利用OOB错误进行评估。
depths = [5, 10, 15, 20]  # 定义树的深度
for depth in depths:
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=depth, oob_score=True, random_state=42)
    rf_clf.fit(X, y)
    plot_decision_boundary(rf_clf, X, y, f'Random Forest Depth {depth} - OOB Score: {rf_clf.oob_score_:.2f}')


# In[ ]:




