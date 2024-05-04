#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

def create_data():
    '''
    X: 一个5行2列的NumPy数组，表示5个样本的特征。每行代表一个样本，每个样本有两个特征。
    y: 一个包含5个元素的NumPy数组，表示每个样本的类标签。这里有3个正类（1）和2个负类（-1）。
    '''
    # 创建一个简单的二分类数据集
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])
    return X, y
'''
X: 输入的数据集。
dimension: 用于分类比较的特征维度（列索引）。
thresh_val: 分类的阈值。
thresh_ineq: 阈值不等式的方向。可以是 "lt" （小于等于时为负类）或 "gt" （大于时为负类）。


ret_array: 初始化为全1的数组，大小与样本数量相同。这表示开始时所有样本都假设为正类。

如果 thresh_ineq 是 "lt"，则当 X 中的特定维度的值小于或等于 thresh_val 时，
将对应的 ret_array 中的值设置为 -1.0，表示这些样本为负类。

如果 thresh_ineq 是 "gt"，则当 X 中的特定维度的值大于 thresh_val 时，进行同样的操作。
'''
def stump_classify(X, dimension, thresh_val, thresh_ineq):
    # 通过阈值比较对数据进行分类
    ret_array = np.ones((np.shape(X)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[X[:, dimension] <= thresh_val] = -1.0
    else:
        ret_array[X[:, dimension] > thresh_val] = -1.0
    return ret_array#即返回每个样本基于阈值条件被分类的结果。


# In[14]:


'''
X: 训练数据的特征矩阵。
y: 训练数据的标签向量。
D: 每个训练样本的权重。
'''
def build_stump(X, y, D):
    m, n = np.shape(X)#数据集的样本数和特征数。
    num_steps = 10.0 #每个特征的阈值将在其最小值和最大值之间分成多少步骤来考察。
    best_stump = {}#用来存储最佳决策树桩的信息。
    best_class_est = np.zeros((m, 1))#存储当前最佳决策树桩的分类结果。
    min_error = np.inf# 初始值假设最小错误率为无穷大，用于记录最小的加权错误率。

    for i in range(n):
        #range_min, range_max: 当前特征的最小值和最大值。
        range_min = X[:, i].min()
        range_max = X[:, i].max()
        #根据num_steps计算出的步长。
        step_size = (range_max - range_min) / num_steps
        #j: 从-1开始，到num_steps + 1结束，允许在最小值和最大值之外一点的探索，以确保覆盖所有可能的阈值。
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:#遍历'lt'（小于）和'gt'（大于），代表不同的不等式。
                thresh_val = (range_min + float(j) * step_size)# 计算出的当前阈值。
                '''
                把数据集、特征维度、阈值、代入'lt'（小于）和'gt'（大于），不同的不等式。
                '''
                predicted_vals = stump_classify(X, i, thresh_val, inequal)
                err_arr = np.ones((m, 1))#错误数组，初始化为1（假设所有分类都是错误的）。
                err_arr[predicted_vals == y.reshape(-1, 1)] = 0#更新错误数组中正确分类的位置为0。
                weighted_error = D.T.dot(err_arr)#计算加权错误，通过点积将错误数组与样本权重D相乘。
                '''
                如果当前决策树桩的加权错误率低于之前的最佳记录，则更新最佳决策树桩信息。
                '''
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est#返回包含最佳决策树桩的字典、最小错误率和最佳分类估计。


# In[15]:


'''
实现AdaBoost算法的核心训练过程，主要用于训练一个由多个弱分类器（决策树桩）组成的强分类器。
数据 X（特征），标签 y，以及算法迭代的次数 num_iters（默认为40次）。
这个函数的核心是通过逐步提升错误分类样本的权重，逐渐增强分类模型的鲁棒性，最终获得较高的分类精度。
'''
def adaboost_train_ds(X, y, num_iters=40):
    weak_class_arr = []#初始化用于存储每一轮迭代中得到的最佳弱分类器及其相关信息。
    '''
    m 是数据集中样本的数量。
    初始化样本权重 D，一开始每个样本的权重都是相等的，即每个样本的初始权重为 1/m。
    初始化一个数组 agg_class_est，用于累加每个弱分类器的加权预测结果，这将用于计算整个模型的最终预测。
    '''
    m = np.shape(X)[0]
    D = np.ones((m, 1)) / m
    agg_class_est = np.zeros((m, 1))

    for i in range(num_iters):
        '''
        调用 build_stump 函数寻找当前权重 D 下的最佳弱分类器 best_stump，
        该函数还返回该分类器的错误率 error 和对训练数据的分类结果 class_est。
        '''
        best_stump, error, class_est = build_stump(X, y, D)
        '''
        根据错误率计算该弱分类器的权重 alpha，使用一个很小的常数 1e-16 防止除零错误。
        权重 alpha 反映了这个弱分类器在最终决策中的重要性，错误率越低，权重越大。
        alpha 的公式就是
        α= 1/2  ​ log(1−ϵ/ ϵ)
        该公式确保了当错误率 𝜖接近 0 时（即学习器表现很好时），该学习器的权重𝛼趋向于无限大，对最终的模型预测影响较大。
        当 𝜖 接近 0.5 时（即学习器表现仅略好于随机猜测时）， 𝛼趋向于 0，意味着该学习器的影响很小。
        当 𝜖 大于 0.5 时（即学习器表现还不如随机猜测时），𝛼可能是负值，表示需要反向调整其对最终预测的贡献。
        '''
        alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-16))
        '''
        将计算得到的 alpha 值存储到 best_stump 字典中，并将其添加到 weak_class_arr 列表中。
        '''
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        '''
        这个公式是用于计算新的样本权重的关键部分。
        计算的是真实标签与分类结果的乘积。如果分类结果正确（即预测值和真实值相同），
        那么结果将为+1（因为 1×1=1    和 (−1)×(−1)=1；
        如果分类结果错误（即预测值和真实值相反），
        那么结果将为-1（因为1×(−1)=−1和 (−1)×1=−1。
        如果一个样本被错误分类（y * class_est 为-1），那么 exp(-alpha * -1) 将增加这个样本的权重，因为 exp(alpha) 总是大于1。
        如果一个样本被正确分类（y * class_est 为+1），那么 exp(-alpha * 1) 将减少这个样本的权重，因为 exp(-alpha) 总是小于1（但大于0）。
        
        这个计算得到的 expon 数值被用于更新每个样本的权重 𝐷。
        D = D * np.exp(expon): 根据 expon 的值，相应地调整每个样本的权重。
        这确保了在下一轮中，被当前分类器错误分类的样本将获得更高的关注（权重更大），
        而被正确分类的样本的权重则减少。也就是说错的权重越大、对的权重越小。
        '''
        expon = -alpha * y.reshape(-1, 1) * class_est
        D = D * np.exp(expon)
        D = D / D.sum()
        #更新累计的类别估计 agg_class_est，并计算当前模型的错误率 error_rate。
        agg_class_est += alpha * class_est
        agg_errors = np.sign(agg_class_est) != y.reshape(-1, 1)
        error_rate = agg_errors.mean()
        #如果错误率达到0，即所有样本都被正确分类，则提前结束迭代。
        if error_rate == 0.0:
            break

    return weak_class_arr, agg_class_est


# In[16]:


X, y = create_data()
weak_class_arr, agg_class_est = adaboost_train_ds(X, y, num_iters=10)
print("Classifiers:", weak_class_arr)


# In[ ]:




