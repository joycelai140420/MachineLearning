#!/usr/bin/env python
# coding: utf-8

# In[6]:


#这段参考Hung-yi Lee老师的上课范例，但要注意的是不同版本呼叫命名方法会有差异，
#有可能是独立Keras 跟TensorFlow 的 Keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D , MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD ,Adam # 注意导入路径
from keras.utils import np_utils
from keras.datasets import mnist
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# In[7]:


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


# In[8]:


(x_train, y_train), (x_test, y_test)=load_data()
print(x_train.shape)#有10000笔数据，分别是784维的vector
#print(x_train[0]) #有值表示这个有没有被涂黑，表示这个pixel颜色有多深，图的最黑=1
#print(y_train[0]) #表示0,1,2,3,4,5,6,7,8,9，有涂黑的是1，其位置就是对应的数字


# In[9]:


models = Sequential()
#优化器和学习率调整、更换损失函数、减少模型复杂度只有3层隐藏层512个神经元
models.add(Dense(units=512, activation='relu', input_dim=28*28))
#增加Dropout层
models.add(Dropout(0.5))
#批量归一化
#models.add(BatchNormalization())
models.add(Dense(units=512, activation='relu'))
models.add(Dropout(0.5))
#批量归一化
#models.add(BatchNormalization())
models.add(Dense(units=10, activation='softmax'))

models.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#更早的停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

models.fit(x_train, y_train, epochs=22, batch_size=100, validation_split=0.1, callbacks=[early_stopping])

train_loss, train_accuracy = models.evaluate(x_train, y_train)
test_loss, test_accuracy = models.evaluate(x_test, y_test)

print(f"\nTraining Loss: {train_loss}, Training Accuracy: {train_accuracy}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[ ]:




