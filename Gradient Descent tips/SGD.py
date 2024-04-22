#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]


# In[2]:


x = np.arange(-200, -100, 1)  # bias
y = np.arange(-5, 5, 0.1)     # weight
z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)

for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        for n in range(len(x_data)):
            z[j][i] += (y_data[n] - b - w * x_data[n]) ** 2
        z[j][i] /= len(x_data)


# In[3]:


b = -120  # initial b
w = -4    # initial w
lr = 0.1  # learning rate for SGD tends to be higher
iteration = 100000
b_history = [b]
w_history = [w]

for i in range(iteration):
    n = np.random.randint(len(x_data))  # select one data point randomly
    x_n = x_data[n]
    y_n = y_data[n]

    # calculate gradients for one selected data point
    b_grad = -2.0 * (y_n - b - w * x_n)
    w_grad = -2.0 * (y_n - b - w * x_n) * x_n

    # update parameters
    b -= lr * b_grad
    w -= lr * w_grad

    b_history.append(b)
    w_history.append(w)

plt.contourf(x, y, z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()


# In[ ]:




