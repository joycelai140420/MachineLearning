Regression（回归分析）和Logistic Regression（逻辑回归）不是一样的，它们是用于不同类型的预测问题的统计方法。


回归分析 (Regression Analysis)
回归分析是一种统计技术，用于建立一个或多个自变量（解释变量）与一个连续依变量（目标变量）之间的关系模型。其目的是预测或解释一个数值型的响应变量。最常见的回归分析是线性回归（Linear Regression），其中模型预测的是一个线性方程。例如，预测房屋价格基于其面积、房间数等因素。

逻辑回归 (Logistic Regression)
逻辑回归，虽然名称中包含“回归”，但实际上是一种用于分类问题的统计方法，特别是二分类问题。它用于预测一个二元依变量（目标变量是两个类别中的一个，通常表示为0和1）的概率。逻辑回归模型输出一个在0到1之间的数值，这个数值通常被解释为属于某个类别的概率。例如，根据病人的各种体征数据来预测是否有心脏病。

主要差异
目标类型：
  线性回归：预测连续的数值型输出。
  逻辑回归：预测二元分类的概率（例如，是/否，0/1）。
输出：
  线性回归：输出一个任意范围内的数值。
  逻辑回归：输出一个在0和1之间的概率值，通过S型函数（Sigmoid function）转换线性方程的输出。
应用：
  线性回归：用于预测那些量度型的数据，如金额、重量、温度等。
  逻辑回归：用于预测事件发生的概率，如是否发生某种疾病、用户是否点击广告等。


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b8c98fdf-3ddb-40d7-a306-31c5b9ff6752)

简易描述差异就是线性回归（Linear Regression）本质上是一系列变量的线性组合再加上偏置项b，而逻辑回归（Logistic Regression） 是在线性回归（Linear Regression） 的基础上加了一层sigmoid函数，将线性函数转变为非线性函数。sigmoid函数的形状呈现为“S”形（如下图所示），它能将任意实数映射到0-1之间的某个概率值上。

tips:
loss function : 
如果我们function是Estimation error 估测误差，就是跟目标值的差异，差越多就是越不好的function，差越多小就是越好的function 。所以要选最小的ˋ差min，用 这个arg min L(w,b)，再用Gradient Descent去解这个function。在用Estimation error作为loss function 后面加上期待值来进行regularization，期待值是为了」让其function 变得平滑，让输入的值对输出的值起伏不容易太大，让输入的杂讯不敏感。例如输入值由outlierts情况就会希望这个function越平滑，且regularization不用考虑b，因为bias跟平滑不相关只跟方向对不对有影响,根据Hung-yi Lee 老师说明是：
$\lambda$ 值越大，代表考慮 smooth 這項的影響力越大，得到的 function 就越平滑
$\lambda$ 值越大，training data 上得到的 error 越大（因為傾向考慮 w 的值而減少考慮 error）
$\lambda$ 值越大，testing data 上的 error 可能會變小 ($\lambda = 100$)，但是 $\lambda$ 太大時，error 又會變大 ($\lambda=1000$)
所以，我們必須調整 $\lambda$ 來決定 function 的平滑程度，找到最小的 testing error。

模型判断是bias大还是variance大，根据Hung-yi Lee 老师说明是：
Underfitting : if your model cannot even fit the training examples,then you have large bias.
Overfitting : if your can fit the training data,but large error on testing data,then you probably have large variance.
如果是large bias，你要做的是redesign model，表示你现在这个model里面可能根本没有包含你的targe。方法可以使修改你的model加入更多feature。或是修正你的function set。找更多的data都是没有帮助。
如果是large variance，要就是增加你的data，或是generate假的training data，根据业务知识点或自身理解制造更多data。例如手写辨识的时候，因为每个人手写到字迹不一样就把training data里面图片左转15度右转15度等等生成不同角度的图片来制造更多training data。或是影像辨识，只有左边开过来的火车，没有右边开过来的火车，就将图片进行翻转。语音辨识例如只有男生的training data没有女生就用变声器转换一下来生成更多training data。如果没有办法增加你的training data，也可以使用Regularzation，就是在loss function 后面加上一个term，让其function越平滑，让你的参数越小越好，如前面说的但也有可能会伤害bias，所以要调整好W让你在bias跟variance取得平衡。

你千万别认为拿3个不同的model来训练你的training set,在testing set表现最好的那个model，就认为也可以训练真实现实其他testing set，因为在真正的testing set上，这个model不见得是最好的。
![1713694742786](https://github.com/joycelai140420/MachineLearning/assets/167413809/1dc70ee9-77bd-4fa5-9c33-03b0a4e80311)

