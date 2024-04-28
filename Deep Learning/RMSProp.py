#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
'''
x: 当前参数的值。
dx: 参数的梯度。
cache: 移动平均的平方梯度，初始化为零矩阵。
lr (Learning Rate): 学习率，控制参数更新的步长。
decay_rate: 控制历史信息在移动平均中的保留程度。
epsilon: 用于数值稳定性的小常数，防止除以零。
'''
def rmsprop_update(x, dx, cache=None, lr=0.01, decay_rate=0.99, epsilon=1e-8):
    """
    A RMSprop update rule implementation using numpy.
    
    Inputs:
    - x: Current value of the parameter.
    - dx: Current gradient of the parameter.
    - cache: Moving average of squared gradients.
    - lr: Learning rate.
    - decay_rate: Decay rate for the moving average of squared gradients.
    - epsilon: Small constant for numerical stability.

    Returns:
    - next_x: Updated parameter value.
    - config: Updated cache with the new moving average of squared gradients.
    """
    if cache is None:
        cache = np.zeros_like(x)
    
    # Update cache with squared gradient
    #更新缓存 ，缓存维护了历史梯度平方的指数衰减平均。
    cache = decay_rate * cache + (1 - decay_rate) * (dx ** 2)
    
    # Update the parameter
    #参数更新:使用缓存值调整学习率并更新参数
    x -= lr * dx / (np.sqrt(cache) + epsilon)
    
    return x, cache


# In[2]:


# Example usage
if __name__ == "__main__":
    # Assume some initial parameters
    x = np.array([1.0, 2.0, 3.0])
    dx = np.array([0.5, -0.5, 1.0])
    
    # Perform one update
    cache = None  # Initially, there's no cache
    learning_rate = 0.1
    decay_rate = 0.9
    epsilon = 1e-8
    
    x_updated, cache_updated = rmsprop_update(x, dx, cache, learning_rate, decay_rate, epsilon)
    #展示了经过一次 RMSProp 更新后的参数值。
    print("Updated parameters:", x_updated)
    #显示了更新后的缓存值，它是平方梯度的衰减平均。
    print("Updated cache:", cache_updated)


# In[ ]:


#in TensorFlow.Keras use RMSProp 
from tensorflow.keras.optimizers import RMSprop
#rho 参数对应于上面的衰减率 𝛾，而 epsilon 是为了数值稳定性添加的小常数。
model.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

