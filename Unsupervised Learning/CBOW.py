#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
以下是一个简化的、未优化的示例，展示如何使用纯 Python 实现一个简单的 CBOW 模型。
这个示例将使用随机梯度下降（SGD）来训练模型，并使用一个非常小的词汇表和数据集来保持简洁。
前后的词汇来预测中间
'''
import numpy as np


# In[4]:


#定义一个简单的词汇表和一些句子。这里每个单词都用一个独热编码表示，是一种简单的向量化方式。
data = {
    'the': [1, 0, 0, 0, 0],
    'quick': [0, 1, 0, 0, 0],
    'brown': [0, 0, 1, 0, 0],
    'fox': [0, 0, 0, 1, 0],
    'jumps': [0, 0, 0, 0, 1],
}
sentences = [
    ['the', 'quick', 'brown'],
    ['quick', 'brown', 'fox'],
    ['brown', 'fox', 'jumps']
]


# In[9]:


#设置学习率、训练周期数、嵌入向量大小和上下文窗口大小。
learning_rate = 0.1
epochs = 100
embed_size = 2  # 嵌入向量的维度
context_size = 2  # 上下文窗口大小


# 初始化权重，这里使用随机数开始，权重将在训练过程中进行调整。
input_layer = np.random.rand(5, embed_size)  # 词嵌入矩阵
print(input_layer)
output_layer = np.random.rand(embed_size, 5)  # 输出层权重
print(output_layer)


# In[10]:


#定义训练函数，循环处理每个句子和每个单词。计算上下文词向量的平均值作为模型的输入。
# 训练函数
def train(epochs, learning_rate, input_layer, output_layer):
    for epoch in range(epochs):
        loss = 0
        for sentence in sentences:
            for i, word in enumerate(sentence):
                # 获取上下文词和目标词
                target_word_vec = np.array(data[word])
                context_words = [sentence[j] for j in range(max(0, i - context_size), min(i + context_size + 1, len(sentence))) if j != i]
                context_word_vecs = np.mean([np.array(data[w]) for w in context_words], axis=0)

                # 前向传播
                #计算隐藏层：将上下文向量与输入层权重相乘。
                hidden_layer = np.dot(context_word_vecs, input_layer)
                #计算输出层：将隐藏层输出与输出层权重相乘。
                output = np.dot(hidden_layer, output_layer)

                # 使用 softmax 计算概率分布，将输出层的分数转换为概率。
                exp_scores = np.exp(output)
                probs = exp_scores / np.sum(exp_scores)

                # 计算损失：交叉熵损失
                loss -= np.log(probs[np.argmax(target_word_vec)])

                # 反向传播
                d_output = probs
                d_output[np.argmax(target_word_vec)] -= 1

                # 更新输出层和输入层权重。
                output_layer -= learning_rate * np.outer(hidden_layer, d_output)
                input_layer -= learning_rate * np.outer(context_word_vecs, np.dot(output_layer, d_output))
        #每10个周期打印一次损失，以监控训练进度。
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# 开始训练模型
train(epochs, learning_rate, input_layer, output_layer)


# In[ ]:




