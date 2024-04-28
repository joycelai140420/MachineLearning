#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def momentum_update(x, dx, v=None, lr=0.01, mu=0.9):
    """
    Momentum update rule implementation using numpy.
    
    Inputs:
    - x: Current value of the parameter.
    - dx: Current gradient of the parameter.
    - v: Current velocity (momentum).
    - lr: Learning rate.
    - mu: Momentum factor.

    Returns:
    - next_x: Updated parameter value.
    - next_v: Updated velocity (momentum).
    """
    if v is None:
        v = np.zeros_like(x)
    
    # Update velocity
    v = mu * v - lr * dx
    
    # Update parameter
    x += v
    
    return x, v


# In[2]:


# Example usage
if __name__ == "__main__":
    # Assume some initial parameters
    x = np.array([1.0, 2.0, 3.0])
    dx = np.array([0.5, -0.5, 1.0])
    
    # Perform one update
    v = None  # Initially, there's no velocity
    learning_rate = 0.1
    momentum_factor = 0.9
    
    x_updated, v_updated = momentum_update(x, dx, v, learning_rate, momentum_factor)
    print("Updated parameters:", x_updated)
    print("Updated velocity:", v_updated)


# In[3]:


from tensorflow.keras.optimizers import SGD

# 初始化优化器，设定学习率和动量
# Keras 的 SGD 优化器允许您指定动量参数momentum
# learning_rate 设置为 0.01，这是步长大小。
# momentum 设置为 0.9，这是动量系数
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# 配置模型，使用上面创建的优化器
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:




