#!/usr/bin/env python
# coding: utf-8

# In[21]:


#这段参考Hung-yi Lee老师的上课范例，但要注意的是不同版本呼叫命名方法会有差异，
#有可能是独立Keras 跟TensorFlow 的 Keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D , MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD ,Adam # 注意导入路径
from keras.utils import np_utils
from keras.datasets import mnist


# In[22]:


def load_data():
    #加载数据集并处理
    (x_train, y_train), (x_test, y_test) = mnist.load_data()#归一化处理，注意必须进行归一化操作，否则准确率非常低，图片和标签
    number=10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train =x_train.reshape(number,28*28)
    x_test = x_test.reshape(x_test.shape[0],28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
    x_train = x_train
    x_test = x_test
    x_train = x_train/255
    x_test=x_test/255
    return (x_train, y_train), (x_test, y_test)


# In[23]:


(x_train, y_train), (x_test, y_test)=load_data()
print(x_train.shape)#有10000笔数据，分别是784维的vector
#print(x_train[0]) #有值表示这个有没有被涂黑，表示这个pixel颜色有多深，图的最黑=1
#print(y_train[0]) #表示0,1,2,3,4,5,6,7,8,9，有涂黑的是1，其位置就是对应的数字


# In[29]:


#搭建神经网络（全连接）
models = Sequential()
# Dense  -> Fully Connected Layer
# 向模型中添加一个Dense层
models.add(Dense(units=633, activation='sigmoid', input_dim=28*28))
#不用在设定input_shape直接会接上面那一段
models.add(Dense(units=633, activation='sigmoid'))
#不用在设定input_shape直接会接上面那一段
models.add(Dense(units=633, activation='sigmoid'))

#不加前Test Acc: 0.1135，再加10层Test Acc还是很差，就可以参考后面范例dnn tips
for i in range(10):
    models.add(Dense(units=633, activation='sigmoid'))
#y输出只有10维
models.add(Dense(units=10, activation='softmax'))

models.compile(optimizer=SGD(learning_rate=0.1), loss='mse', metrics=['accuracy'])
#models.compile(optimizer=SGD(lr=0.1), loss='mse', metrics=['accuracy'])

models.fit(x_train, y_train, epochs=22, batch_size=100 )  

#假设train完要看一下最后的评估结果
result=models.evaluate(x_test,y_test)
print('\nTest Acc:',result[1])


# In[30]:


'''
这边我先将上诉的范例添加validation功能
然后做完N-fold cross Validation Average Test Acc: 0.20644002
可以发现做validation功能会提高准确性
比没做提高一些，可以参考后面范例dnn tips，再提高Acc。
'''
'''
利用models.fit自动做validation,这个参数允许你指定一个比例，
从训练数据中自动分割出一部分作为验证数据。这是最简单的方法来进行验证，
但它不是真正的 N-fold 交叉验证，因为它只分割一次数据。
# 在模型训练时指定 validation_split
models.fit(x_train, y_train, epochs=22, batch_size=100, validation_split=0.1)

'''
'''
利用实现N-fold cross Validation
为了实现 N-fold 交叉验证，我们需要手动分割数据并循环执行训练和验证过程。
这里我们可以使用 scikit-learn 的 KFold 类来帮助我们分割数据。
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import KFold

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

n_splits = 5
'''
使用 KFold 来生成训练和验证索引，然后在每次循环中使用这些索引来选择数据，构建模型，
并进行训练和评估。最后，它计算所有循环中测试精度的平均值。
'''
kf = KFold(n_splits=n_splits)

accuracies = []
for train_index, val_index in kf.split(x_train):
    x_train_k, x_val_k = x_train[train_index], x_train[val_index]
    y_train_k, y_val_k = y_train[train_index], y_train[val_index]

    model = Sequential([
        Dense(633, activation='sigmoid', input_dim=28*28),
        Dense(633, activation='sigmoid'),
        Dense(633, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.1), loss='mse', metrics=['accuracy'])
    model.fit(x_train_k, y_train_k, epochs=22, batch_size=100, validation_data=(x_val_k, y_val_k))
    score = model.evaluate(x_test, y_test)
    accuracies.append(score[1])

print('\nAverage Test Acc:', np.mean(accuracies))


# In[ ]:




