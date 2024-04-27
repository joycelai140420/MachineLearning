![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3dda1b36-5e31-4bcc-8bdf-3d4ddd29602a)DNN初探笔记

全连接前馈神经网络（Fully Connected Feedforward Network）
也称为多层感知机（MLP），是深度学习中最基础的一种网络结构。这种网络结构包括一个输入层、若干个隐藏层以及一个输出层。每一层由多个神经元组成，每个神经元与前一层的所有神经元相连接（全连接）。怎么连接是手动设计，

原理简述

网络结构：网络由多层组成，其中每一层都是前一层的线性变换（加ｗ加ｂ）后通过一个非线性激活函数处理的结果。

激活函数：通常使用ReLU、Sigmoid或Tanh等函数（不过现在比较少再用Sigmoid后面会解释），增加网络的非线性，使其能够学习更复杂的函数映射。

前向传播：数据从输入层进入，经过各隐藏层的处理，最后通过输出层得到预测结果。

反向传播与梯度下降：使用反向传播算法计算损失函数关于每个权重的梯度，然后用梯度下降（或其他优化算法）更新权重，以最小化损失函数。

应用

图像识别：用于识别数字、物体、人脸等。

语音识别：转换语音为文本，应用于虚拟助手和自动字幕生成。

文本分类：如情感分析、主题分类等。

游戏玩法：如AlphaGo等AI程序，使用神经网络决定行动策略。

优点

强大的表示能力：理论上，一个足够大的网络可以近似任何复杂的函数。

广泛的应用领域：从视觉到自然语言处理，前馈网络都有广泛应用。

结构简单：架构直观，易于构建和修改。

缺点

过拟合：容易在小数据集上过拟合。

计算密集：尤其是当网络很深或很宽时。

需要大量的数据：为了表现良好，通常需要大量数据来训练。

后面请参考 台大Hung-yi Lee 课程内容

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/127bd020-1431-4a95-8604-397d506c08c7)

Step 1: Define a Neural Network


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7cbe0c0d-9311-4c80-a5dd-6eae2c30b005)

「Deep」意即 Many hidden layers

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e1d01321-48b0-4192-816a-f65803762b7e)

![1714186450537](https://github.com/joycelai140420/MachineLearning/assets/167413809/7bdafae9-d91a-4230-99f5-15a2f44ffc6d)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4e50c907-165e-4113-bd1e-2a815cdb9273)


Output layer 即為 Multi-Class Classifier

將 hidden layer 視為 feature extractor 將 output layer 視為 multi-class classifier，最後一個 layer 會加上 Softmax function

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/43c6c715-5470-4e10-bb9a-d720905edc2d)

Step 2: Goodness of function

決定參數的好壞，計算 output (y) 跟目標 y^ 之間的 cross entropy，調整 network 的參數，讓 cross entropy 越小越好。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2a13f647-521c-4bc0-93df-a1414237f50f)

Step 3: Pick the best function
將所有 data 的 cross entropy 全部加起來的總和，得到 total loss (L)。在 function set 中找一個 function，或是找一組 network 的 parameter，讓 total loss 越小越好。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/18214e0e-5904-4261-a780-ddb6053231fc)

你可以参考The_first_DNN.py，一个用toolkit，一个只用numpy写的DNN


DNN tips:

Batch & Epoch：

做 Deep Learning 時，會將 training data 隨機的選 x 個放進一個 batch，x 即為 batch_size

不斷地 pick batch，直到所有的 mini batch 都被 update 一次，即為一個 epoch

nb_epoch 就是重複 update 幾次 epoch

本例中， batc_size = 100， nb_epoch = 20 1 個 epoch 中會 update 100 次參數，20 個 epoch 即總共會 update 2000 次參數

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f839bfe0-b03c-4376-8123-a3f1ce13208d)


Speed Comparison（實作上的比較）

Stochastic Gradient Descent (Batch size = 1) v.s. Mini-batch (Batch size > 1)

如下圖，實際在 GTX980 跑 MNIST 的 50000 個 examples 時

batch size = 1：一個 epoch 跑166s batch size = 10：一個 epoch 跑 17s

亦即，batch size 越大，每算一個 epoch 的時間越快(短) 故，在同樣時間內，參數 update的數目幾乎相同 故，選擇 batch size 較大者，因為較 穩定

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e09ccbef-d115-4e6b-95cc-8bbeb15c5524)

那為何不將 batch size 設非常大呢？

硬體限制，GPU 將無法平行運算

容易陷入 saddle point 或 local minimum 中 (它的 error surface 是坑坑洞洞的)
以矩陣運算解釋，如下圖

Stochastic Gradient Descent：黃色和綠色 z^1 依序和 W^1 做運算 Mini-batch：黃色和綠色 z^1 同時和 W^1 做運算

可明顯看出 Mini-batch 運算速度較快

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a17b323f-c073-498a-be71-52202e027750)


