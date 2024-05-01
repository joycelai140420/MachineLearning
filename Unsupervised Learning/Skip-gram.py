#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
以下是一个简化的、未优化的示例，展示如何使用纯 Python 实现一个简单的 Skip-gram 模型。
中间的词汇来预测前后
'''
import random
import math

# 示例文本
corpus = "the quick brown fox jumps over the lazy dog"

# 构建词汇表
words = list(set(corpus.split()))
word_to_idx = {w: idx for idx, w in enumerate(words)}
idx_to_word = {idx: w for idx, w in enumerate(words)}

# 超参数
learning_rate = 0.01
epochs = 100
embed_size = 2
window_size = 1

# 初始化权重
input_layer = {word: [random.uniform(-1, 1) for _ in range(embed_size)] for word in words}
output_layer = {word: [random.uniform(-1, 1) for _ in range(embed_size)] for word in words}

# 生成训练数据
'''
具体来说，它根据给定的窗口大小（window_size），
为每个词生成与其相邻的词对（中心词和上下文词对）。
'''
def generate_training_data(words, window_size):
    data = []# 初始化一个空列表，用来存储生成的词对
    for idx, word in enumerate(words): # 遍历每个单词及其索引
         # 生成上下文环境中的单词
        for neighbor in words[max(idx - window_size, 0): min(idx + window_size + 1, len(words))]:
            # 确保上下文中的单词不是中心词本身
            if neighbor != word:
                data.append((word, neighbor)) # 将中心词和上下文词的组合作为一个元组添加到数据列表中
    return data

training_data = generate_training_data(corpus.split(), window_size)


# In[6]:


# Softmax 函数
'''
这一步的目的是将原始的输出向量转换成一个有效的概率分布，其中每个元素的值都在0到1之间，且所有元素之和为1。
这样的输出可以被解释为概率，非常适合分类任务中表示不同类别的预测概率。
'''
def softmax(x):
    e_x = [math.exp(i) for i in x]# 对输入向量x中的每一个元素计算其指数
    return [ex / sum(e_x) for ex in e_x]# 将计算得到的指数值标准化，使所有元素之和为1


# In[8]:


# 训练模型
def train(data, epochs, learning_rate):
    for epoch in range(epochs):
        loss = 0
        for input_word, target_word in data:
            input_vec = input_layer[input_word]
            target_vec = output_layer[target_word]

            # 计算预测值和实际值之间的误差
            preds = softmax([sum([iv * ov for iv, ov in zip(input_vec, output_vec)]) for output_vec in output_layer.values()])
            target_idx = word_to_idx[target_word]
            error = [-preds[idx] + (1 if idx == target_idx else 0) for idx in range(len(words))]

            # 更新权重
            for idx, ov_key in enumerate(output_layer):
                ov = output_layer[ov_key]
                err = error[idx]  # 定位具体的错误值
                input_updates = [learning_rate * err * ov_i for ov_i in ov]
                output_updates = [learning_rate * err * iv for iv in input_vec]

                input_layer[input_word] = [iv + up for iv, up in zip(input_layer[input_word], input_updates)]
                output_layer[ov_key] = [ov_i + up for ov_i, up in zip(output_layer[ov_key], output_updates)]

            # 计算损失
            loss -= math.log(preds[target_idx])

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss / len(data)}')

# 开始训练
train(training_data, epochs, learning_rate)


# In[ ]:




