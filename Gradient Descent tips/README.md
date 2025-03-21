梯度下降（Gradient Descent）简介

梯度下降是一种优化算法，主要用于找到函数的局部最小值。在机器学习中，它通常用于优化损失函数，即调整模型参数以最小化误差或预测与真实值之间的差距。基本思想是：从一个随机的参数值开始，逐步向参数空间的梯度（即最陡峭的上升或下降方向）的反方向迈进，以期望达到损失函数的最小值。


梯度下降的优点

通用性：适用于各种可导函数的优化问题，广泛用于各类参数优化场景。

实现简单：算法结构清晰，易于编码实现，是许多复杂学习算法的基础。

灵活性：通过调整学习率和迭代次数，可以适应不同的数据规模和精度需求。

梯度下降的缺点

收敛速度：对于非凸函数，梯度下降可能只能找到局部最小值而非全局最小值。

参数敏感：学习率和初始参数的选择可以显著影响算法的性能和收敛速度。

计算成本：在大规模数据集上，每次迭代需要计算所有数据点的梯度，可能导致计算量很大。

应用

机器学习：在几乎所有需要训练和优化参数的机器学习模型中，如线性回归、逻辑回归、神经网络等。

深度学习：深度神经网络的训练通常依赖于梯度下降的变种，如随机梯度下降（SGD）、Adam等。

数据科学：在预测分析和特征工程中优化模型参数。

人工智能：在强化学习等领域中用于优化策略或决策模型。

梯度下降的工作原理

在执行时，梯度下降会计算当前参数值下的损失函数梯度，然后调整参数朝梯度的反方向（即下降最快的方向）移动一定步长，这个步长通常由学习率决定。通过迭代这一过程，梯度下降逐渐驱使参数向使损失函数值最小化的方向调整，直到达到收敛条件或完成设定的迭代次数。

通过上述描述，可以看出梯度下降是一种基于导数的优化方法，主要目的是寻找函数的最小值点，广泛应用于机器学习和人工智能领域中的模型训练和参数调优。

tips:参考台大Hung-yi Lee 课程内容

learning rate调太小，跑得慢，调太大可能找不到local minimum，或是直接飞出去，但观察loss 参数如果是高维度是没有办法visualize，但是你可以 visualize参数的变化对这个loss的变化去观察learning rate应该怎么调整。如下图：
![1713706519784](https://github.com/joycelai140420/MachineLearning/assets/167413809/9ac56635-c15c-4f43-93c3-622e121181b4)
但也有自动的方法调整learning rat，其中简单方法就是下面这个方法。

Adaptivt learning rate: 
AdaGrad的方法就是把每一个learning rat都除上之前算出来的微分值的root mean square，请参考在Regression的代码 Regression model's Gradient Descent and AdaGrad demo.py ，不过有个问题就是最后的update参数会越来越慢，但也可以用另一个方法是Adam。

Stochastic Gradient Descent（SGD） :會比較快，其原理是只取n個example 做Gradient Descent ，如果data少就不用。随机梯度下降的核心优势在于每次迭代计算速度快，因为它每次只处理一个训练样本，从而快速并频繁地更新模型参数。这通常会使得收敛过程更加快速，在处理大规模数据集时特别有效。然而，SGD的缺点是更新过程中的高方差，可能导致解的质量波动比较大。但是要注意在NN使用SGD时，未必比较快。如下图。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e432fa48-73f1-4d9d-90f2-4a98e4e6394f)


Feature Scaling :

就是先将input data 做一下Scaling，就如同台大Hung-yi Lee 课程投影片说，左圖：長橢圓的 error surface 需要不同的 learning rate，也就是要用 adaptive learning。右圖：正圓形的 error surface，不論從哪個點開始，都會向著圓心走。有做 feature scaling，則在參數的 update 上較有效率。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e1d391eb-e1d8-49ed-b6f2-85b480873b1f)
常见方法有下面图是介绍
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1e4901d2-a862-45fd-b827-53c24720c9be)
代码范例可以参考Z-Score Normalization and Min-Max Scaling.py代码



