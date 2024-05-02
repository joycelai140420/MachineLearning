#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 模拟数据和属性
#创建模拟数据集，其中包括三个已知类别和一个未见类别。每个类别都有一组属性。
def load_data():
    # 假设我们有三个已知类（训练）和一个未知类（测试）
    classes = ['cat', 'dog', 'horse']  # 已知类
    zero_shot_class = ['zebra']  # 零样本类
    
    # 模拟属性：[条纹, 四腿, 尾巴, 宠物]
    attributes = {
        'cat': [0, 1, 1, 1],
        'dog': [0, 1, 1, 1],
        'horse': [0, 1, 1, 0],
        'zebra': [1, 1, 1, 0]
    }
    
    # 创建数据集
    x_train = np.array([attributes[cls] for cls in classes])
    y_train = np.array(classes)
    
    x_test = np.array([attributes[zero_shot_class[0]]])
    y_test = np.array(zero_shot_class)
    
    return x_train, y_train, x_test, y_test

# 构建模型
#建立一个简单的神经网络模型，用于学习属性与类别之间的关系。
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(len(np.unique(y_train)), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

x_train, y_train, x_test, y_test = load_data()
le = LabelEncoder()
y_train_encoded = to_categorical(le.fit_transform(y_train))

model = build_model((x_train.shape[1],))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型,识别已知类别。
model.fit(x_train, y_train_encoded, epochs=100, verbose=2)

# 用训练好的模型预测零样本类
predictions = model.predict(x_test)
predicted_class = le.inverse_transform([np.argmax(predictions)])

print(f"Predicted class for zebra: {predicted_class[0]}")

# 可视化属性
attributes = ['stripes', 'four_legs', 'tail', 'pet']
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(attributes))
bar1 = ax.bar(index, x_train[0], bar_width, label='Cat')
bar2 = ax.bar(index + bar_width, x_test[0], bar_width, label='Zebra')

ax.set_xlabel('Attributes')
ax.set_ylabel('Presence')
ax.set_title('Attributes comparison between Cat and Zebra')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(attributes)
ax.legend()

plt.show()


# In[ ]:




