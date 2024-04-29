#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

# 创建一个虚构的分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)

# 将数据集划分为标记和未标记的数据
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.5, random_state=42)

# 用标记的数据训练一个分类器
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
test_loss = log_loss(y_test, y_prob)
test_accuracy = accuracy_score(y_test, y_pred)
test_error_rate = 1 - test_accuracy

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Error Rate: {test_error_rate:.4f}')

# 基于熵的正则化处理
def entropy_regularization(predictions):
    entropy = -np.sum(predictions * np.log(predictions + 1e-15), axis=1)
    return np.mean(entropy)

# 对未标记的数据进行预测
unlabeled_predictions = classifier.predict_proba(X_unlabeled)

# 计算未标记数据的熵
unlabeled_entropy = entropy_regularization(unlabeled_predictions)

print(f'Unlabeled Data Entropy: {unlabeled_entropy:.4f}')


# In[ ]:




