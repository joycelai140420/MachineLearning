#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
#范例代码出处 Hung-yi Lee 老师的教学范例，其中文注解是我后续添加
#数字后面接句点（.）通常表示这是一个浮点数（float）。
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
#ydata=b+w*xdata


# In[8]:


#创建了偏见b和权重w的参数空间，并初始化一个零矩阵z来存储每对(b, w)的损失函数值。np.meshgrid用于生成一个网格，这在绘制等高线图时非常有用。
x=np.arange(-200,-100,1) #bias
y=np.arange(-5,5,0.1) #weight
z=np.zeros((len(x),len(y)))
X,Y=np.meshgrid(x,y)
#计算损失函数，遍历所有b和w的组合，计算每对参数下的均方误差(MSE)。
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[j]
        z[j][i]=0
        for n in range(len(x_data)):
            z[j][i]=z[j][i]+(y_data[n]-b-w*x_data[n])**2
        z[j][i]=z[j][i]/len(x_data)


# In[11]:


#ydata=b+w*xdata
#设置梯度下降的初始参数、Learning rate和迭代次数。
b=-120 #initial b
w=-4 #initial w
#Learning rate==0.0000001会发现离目标点还有段距离，就表示步伐太小，可以设大一点，但设太大步伐太大又找不到最低点，怎么设都没办法到达目标点
#就看下段代码讲b跟w客值不同Learning rate
lr=0.0000001 
iteration =100000

#store initial values for platting.
b_history =[b]
w_history =[w]

#以下是梯度下降迭代
#Iterations
for i in range(iteration):
    #初始化梯度: b_grad 和 w_grad 分别初始化为 0.0
    b_grad =0.0
    w_grad =0.0
    for n in range(len(x_data)):
        #梯度计算公式 -2(y_data[n] - b - w * x_data[n]) 来自于均方误差的偏导数，
        #这里乘以 2 是因为均方误差函数对 b 和 w 的导数是线性的，而加负号是因为我们进行的是梯度下降。
        #b_grad: 损失函数对偏置 b 的梯度。
        #w_grad: 损失函数对权重 w 的梯度。
        #1.0 表示这里的 x 值对于偏置的导数（因为 b 的系数是 1）。
        b_grad=b_grad-2.0*(y_data[n]-b-w*x_data[n])*1.0
        #因为损失函数中 w 的影响通过 xn传递（即 w 对损失的影响与xn成比例）。
        w_grad=w_grad-2.0*(y_data[n]-b-w*x_data[n])*x_data[n]
        #梯度的这种差异反映了参数 𝑏 和 𝑤 对损失函数的不同影响方式。对于偏置 𝑏，它直接影响预测结果，而与输入特征 𝑥𝑛无关，
        #因此其梯度不涉及 𝑥𝑛。而对于权重 𝑤它通过与每个输入特征 𝑥𝑛 的乘积来影响预测结果，因此在计算 𝑤 的梯度时需要将每个特征值 𝑥𝑛 考虑进去，
        #以正确地表示 𝑤 对损失的具体贡献。
    
    #updata parameters
    b = b -lr * b_grad
    w = w -lr * w_grad
    
    #store parameters for platting
    b_history.append(b)
    w_history.append(w)

#plot the figure
plt.contourf(x,y,z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
#标记最优点
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()


# In[ ]:


#ydata=b+w*xdata
#设置梯度下降的初始参数、Learning rate和迭代次数。
b=-120 #initial b
w=-4 #initial w
#Learning rate==0.0000001会发现离目标点还有段距离，就表示步伐太小，可以设大一点，但设太大步伐太大又找不到最低点，怎么设都没办法到达目标点
#就看这里代码如何b跟w客制不同Learning rate 就是AdaGrad
lr=1 
iteration =100000

#store initial values for platting.
b_history =[b]
w_history =[w]

#以下是梯度下降迭代
#Iterations
for i in range(iteration):
    #初始化梯度: b_grad 和 w_grad 分别初始化为 0.0
    b_grad =0.0
    w_grad =0.0
    for n in range(len(x_data)):
        #梯度计算公式 -2(y_data[n] - b - w * x_data[n]) 来自于均方误差的偏导数，
        #这里乘以 2 是因为均方误差函数对 b 和 w 的导数是线性的，而加负号是因为我们进行的是梯度下降。
        #b_grad: 损失函数对偏置 b 的梯度。
        #w_grad: 损失函数对权重 w 的梯度。
        #1.0 表示这里的 x 值对于偏置的导数（因为 b 的系数是 1）。
        b_grad=b_grad-2.0*(y_data[n]-b-w*x_data[n])*1.0
        #因为损失函数中 w 的影响通过 xn传递（即 w 对损失的影响与xn成比例）。
        w_grad=w_grad-2.0*(y_data[n]-b-w*x_data[n])*x_data[n]
        #梯度的这种差异反映了参数 𝑏 和 𝑤 对损失函数的不同影响方式。对于偏置 𝑏，它直接影响预测结果，而与输入特征 𝑥𝑛无关，
        #因此其梯度不涉及 𝑥𝑛。而对于权重 𝑤它通过与每个输入特征 𝑥𝑛 的乘积来影响预测结果，因此在计算 𝑤 的梯度时需要将每个特征值 𝑥𝑛 考虑进去，
        #以正确地表示 𝑤 对损失的具体贡献。
    
    lr_b=lr_b+b_grad **2
    lr_w=lr_w+w_grad **2
    #updata parameters
    b = b -lr/np.sqrt(lr_b) * b_grad
    w = w -lr/np.sqrt(lr_w) * w_grad
    
    #store parameters for platting
    b_history.append(b)
    w_history.append(w)

#plot the figure
plt.contourf(x,y,z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
#标记最优点
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()


# In[ ]:




