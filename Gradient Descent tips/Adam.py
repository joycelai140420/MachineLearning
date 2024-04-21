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
lr = 0.001
iteration = 100000
b_history = [b]
w_history = [w]

m_b = 0
m_w = 0
v_b = 0
v_w = 0
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad -= 2.0 * (y_data[n] - b - w * x_data[n])
        w_grad -= 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    # Update moving averages of the gradients
    m_b = beta1 * m_b + (1 - beta1) * b_grad
    m_w = beta1 * m_w + (1 - beta1) * w_grad
    v_b = beta2 * v_b + (1 - beta2) * (b_grad ** 2)
    v_w = beta2 * v_w + (1 - beta2) * (w_grad ** 2)

    # Compute bias-corrected first and second moment estimates
    m_b_hat = m_b / (1 - beta1 ** (i + 1))
    m_w_hat = m_w / (1 - beta1 ** (i + 1))
    v_b_hat = v_b / (1 - beta2 ** (i + 1))
    v_w_hat = v_w / (1 - beta2 ** (i + 1))

    # Update parameters
    b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

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




