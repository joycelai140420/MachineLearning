#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import datasets
#从 scikit-learn 中导入 GaussianMixture，这是高斯混合模型，可以用来进行生成模型的学习和数据点的软聚类。
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#StandardScaler，用于特征标准化处理，即将特征值转换为均值为0、方差为1的分布。
from sklearn.preprocessing import StandardScaler

# 加载公开的数据集，以iris数据集为例
iris = datasets.load_iris()
#特征赋值给 X，标签赋值给 y。
X, y = iris.data, iris.target

# 标准化数据，初始化 StandardScaler 对象，并使用 fit_transform 方法对特征 X 进行标准化处理。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集，保留一部分标记数据，其余作为未标记数据
'''
利用 train_test_split 方法将标准化后的数据集分为标记和未标记的数据集。
80%的数据作为未标记数据，剩余20%作为标记数据。
stratify=y 确保划分后的数据集中各类的比例与原始数据集相同。
'''
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_scaled, y, test_size=0.8, stratify=y)

# 初始化GMM
# 假设有3个高斯分布（iris数据集有3类）
gmm = GaussianMixture(n_components=3, random_state=42)

# 训练GMM
# 只用标记数据初始化GMM参数
'''
为 GMM 模型的均值参数 means_ 手动设置初始值，
使用有标记数据的均值，然后在整个数据集上进行拟合。
'''
gmm.means_ = np.array([X_labeled[y_labeled == i].mean(axis=0) for i in range(3)])
gmm.fit(X_scaled)

# 对未标记数据进行伪标记
'''
使用训练好的 GMM 模型对未标记的数据 X_unlabeled 进行预测，生成伪标签。
'''
pseudo_labels = gmm.predict(X_unlabeled)

# 将伪标记数据和标记数据结合在一起
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.hstack((y_labeled, pseudo_labels))

# 训练监督模型
# 使用全部数据重新训练一个GMM
'''
重新初始化一个 GMM 模型，并在合并后的数据集上进行训练，
这是半监督学习的关键步骤，它将未标记数据的信息纳入了模型学习过程中。
因为要算出你的类别的几率，所以n_components要设定你的标记有几类
'''
gmm_final = GaussianMixture(n_components=3, random_state=42)
gmm_final.fit(X_combined)

# 在测试集上评估模型
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, stratify=y_combined)

'''
使用最终的 GMM 模型对测试集进行预测，计算准确率、错误率和损失。
注意，GMM 的 score_samples 返回的是对数似然，对其取负值可以得到平均损失。
'''
y_pred = gmm_final.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
loss = -gmm_final.score_samples(X_test).mean()

print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("Error Rate: ", error_rate)
print(classification_report(y_test, y_pred))


# In[ ]:




