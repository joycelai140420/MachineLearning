DNN初探笔记

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

请参考 台大Hung-yi Lee 课程内容

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7cbe0c0d-9311-4c80-a5dd-6eae2c30b005)

「Deep」意即 Many hidden layers

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e1d01321-48b0-4192-816a-f65803762b7e)

![1714186450537](https://github.com/joycelai140420/MachineLearning/assets/167413809/7bdafae9-d91a-4230-99f5-15a2f44ffc6d)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4e50c907-165e-4113-bd1e-2a815cdb9273)


Output layer 即為 Multi-Class Classifier

將 hidden layer 視為 feature extractor 將 output layer 視為 multi-class classifier，最後一個 layer 會加上 Softmax function

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/43c6c715-5470-4e10-bb9a-d720905edc2d)





