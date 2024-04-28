首先这篇很长，等我有空在分门别类...


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

你可以参考Keras_helloworld_DNN.py 其中范例是参考台大Hung-yi Lee 老师课堂范例，另外我再添加实现N-fold cross Validation范例


如何知道自己的DNN出了什么问题以及如何有方向的改进，以下是来自于台大Hung-yi Lee 老师课程内容

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/32799bbe-47bc-4e1b-a085-cbeab2017e8c)

訓練一個 deep learning 的 network 的流程，應該是由以下三個步驟組成： 定義 function set 和網路架構、決定 loss function、用 Gradient Descent 進行 optimization。 做完以上的流程以後，會得到一個訓練好的 neural network。

接下來，第一件要檢查這個 neural network 在 training set 上有沒有得到好的結果。注意並不是檢查 testing set，而是要先檢查的是這個 neural network 在 training set 上，有沒有得到好的結果。如果沒有的話就必須回頭去檢查，在這三個步驟之中是不是哪邊出了問題。必須思考應該做怎樣的修改，才能在 training set 上得到好的結果。

這邊提到先檢查 training set 的表現，其實是深度學習一個非常獨特的地方。如果今天使用的是其他的方法，比如說 k-nearest neighbor 或 decision tree，在做完以後，其實不太有必要去檢查 training set 的結果，因為在 training set 上的正確率就是 100%，沒有什麼好檢查的。

所以，有人會說 deep learning 的 model 裡面這麼多參數，感覺很容易 overfitting 的樣子。但其實 deep learning 的方法，並不容易 overfitting。所謂的 overfitting 就是在 training set 上表現很好，但 testing set 上表現沒有那麼好。而上面提到的 k-nearest neighbor 和 decision tree，它們的 training set 上正確率都是 100%，這樣才算是非常容易 overfitting。

而對 deep learning 來說，overfitting 往往不是第一個會遇到的問題。這邊並不是說 deep learning 沒有 overfitting 的問題，而是說第一個會遇到的問題，會發生在 training 的時候。它並不是像 k-nearest neighbor 這種方法一樣，一進行訓練就可以得到非常好的正確率。它有可能在 training set 上，根本沒有辦法得到不錯的正確率。所以，這時就要回頭去檢查，在前面的步驟裡面，要做什麼樣的修改，才能在 training set 上得到不錯的正確率。

假設現在，已經在 training set 上得到好的表現，接下來才是把 network 套用在 testing set 上。其實要用 deep learning 在 training set 上得到 100% 的正確率沒有那麼容易，但可能你已經在 MNIST 上得到一個 99.8% 的正確率。而 testing set 上的表現才是大家最後真正關心的表現。

那把它套用到 testing set 上，這個神經網路會在 testing set 上有怎麼樣的表現呢？如果得到的是不好的結果，這種情形才是 Overfitting。在 training set 上得到好的結果，但在 testing set 上得到的是不好的結果，這種情形才能稱之為 Overfitting。

這時就要回頭用一些技巧試著去解決 overfitting 的問題。但有時想要用新的技巧去解決 overfitting 的問題時，反而會讓 training set 上的結果變壞。所以在做這一步的修改以後，還是要先檢查 training set 上的結果。如果 training set 上的結果變壞的話，就要從頭去對訓練的過程做一些調整。那如果同時在 training set 還有自己的 testing set 都得到好的結果的話，就可以把系統真正實際應用，這時就大功告成了。

這邊必須提到一個重點，不要看到所有不好的表現，就說是 overfitting。舉例來說，以下是文獻上的圖。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b454cc88-6d32-4dc2-b631-c37d803d566a)

但在現實訓練中，也常常能夠觀察到類似的情況：右邊這張圖是在 testing set 上的結果。橫坐標是 model 在 Gradient Descent 時參數更新的次數；縱座標是 error rate，越低越好。在這張圖表中，20 層的 network 以黃線表示，而 56 層的 network 以紅線表示。可以觀察到 56 層的network 的 error rate 比較高，它的表現比較差；而 20 層的 neural network，它的表現比較好。

有些人看到這張圖，就會得到一個結論：「56 層 model 的參數太多了。56 層果然沒有必要，這是 overfitting。」但是，真的能夠這樣說嗎？其實在說是 overfitting 之前，必須要先檢查一下在 training set 上的結果。對某些方法來說，不用檢查這件事，像剛剛提到的 k-nearest neighbor 或 decision tree。但是對 neural network 就必須檢查，因為有可能在 training set 上得到的結果是像左圖一樣。橫軸一樣是參數更新的次數，縱軸是 error rate。如果比較 20 層的 neural network 跟56 層的 neural network 的話，會發現在 training set 上 ，20 層的 neural network 的表現本來就比 56 層好。56 層的 neural network 的表現其實是比較差的。

那為甚麼會發生這種情形呢？在訓練 neural network 的時候，有太多問題可能導致訓練結果不好。比如說，local minimum、saddle point，或者是 plateau 的問題。所以有可能這個 56 層的 neural network 在訓練的時候，它就卡在 local minimum，因此得到了一個較差的參數。這並不是 overfitting，而在訓練時就沒有訓練好。有人會說這種情況稱為 underfitting，但是這只是名詞定義的問題。我認為 underfitting 的意思應該是這個 model 的複雜度不足，或者說這個 model 的參數不夠多，所以它的能力不足以解出這個問題。

對這個 56 層的 neural network 來說，雖然它得到比較差的表現，但假如這個 56 層的 network其實是在 20 層的 network 後面另外加上 36 層的 network，那它的參數其實是比 20 層的 network 還多的。所以理論上，20 層的 network 可以做到的事情，56 層的 network 一定也可以做到。假設這個 56 層的 network，前面 20 層就做跟這個 20 層 network 完全相同，後面那 36 層就甚麼事都不做，當成是 identity matrix，那明明可以做到跟 20 層一樣的事情，為甚麼會做不到呢？

但是就是會有很多的問題使得這個 network 沒有辦法做到。56 層的 network 比 20 層的差，並不是因為它能力不夠，因為它只要前 20 層都跟 20 層的一樣，後面都是 identity 明明就可以跟 20 層一樣好，但它卻沒有得到這樣的結果。所以說它的能力其實是足夠的，所以我認為這不是 underfitting，就只是沒有訓練好，而我還不知道有沒有名詞專門指稱這個問題。

所以如果在 deep learning 的文獻上看到一個方法，永遠都必須要思考一下這個方法是要解決什麼樣的問題。因為在 deep learning 裡面有兩個問題：一個是 training set 上的表現不好，一個是 testing set 上的表現不好。當有一個方法被提出的時候，往往就是針對這兩個問題的其中一個處理。舉例來說，等一下會提到一個叫做 dropout 的方法。dropout 也許許多人或多或少都會知道，它是一個很有 deep learning 特色的方法。


