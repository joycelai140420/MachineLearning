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

你可以参考Keras_helloworld_DNN.py 其中范例是参考台大Hung-yi Lee 老师课堂范例，另外我再添加实现N-fold cross Validation范例，可以看出做了N-fold效果会比没做的好，但这个范例是随手写，所以是没train好要改一下模型，必須回頭去檢查「定義 function set 和網路架構、決定 loss function、用 Gradient Descent 進行 optimization」，在這三個步驟之中是不是哪邊出了問題。


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

於是很多人就會說：「所以我今天只要看到 performance 不好，就可以決定要用 dropout。」但是，只要仔細思考一下 dropout 是甚麼時候用的，就會發現 dropout 是在 testing 的結果不好的時候才會使用的，而 testing data 結果好的時候是不會使用 dropout 的。所以如果今天問題是 training 的結果不好，而還是使用 dropout，只會越訓練越差而已。所以，不同的方法處理甚麼樣不同的問題，是必須要想清楚的。

我們剛才提到 deep learning 的流程裡面，在訓練的時候有兩個問題。所以接下來我們會對這兩個問題分開來討論，介紹在遇到這兩個問題的時候，有什麼解決方法

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/776e70a3-ec0a-4fea-83e1-5d2c868e25b0)

首先，如果在 training data 上的結果不好時，可以檢查一下是不是在 network 架構設計時設計得不好。舉例來說，可能 model 使用的 activation function 是比較不好的 activation function，或者說是對 training 比較不利的 activation function。可能可以透過換一些新的 activation function，得到比較好的結果。

我們知道，在 1980 年代的時候，比較常用的 activation function 是 sigmoid function。我們之前有稍微解釋為甚麼要使用 sigmoid function。今天如果我們使用 sigmoid function，其實你可能會發現越深的網路表現不一定比較好。

下圖是在 MNIST 手寫數字辨識上面的結果。當 layer 越來越多的時候，準確率一開始持平後來就變低了；當 layer 是 9 層、10 層時整個結果就崩潰了。有些人看到這張圖，就會說「9 層、10 層參數太多了，overfitting」如之前所說，這種情況並不是 overfitting。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/dd7ff6dd-35a4-4f70-8c90-508957a10b8d)

為甚麼呢？首先要檢查表現不好是不是來自於 overfitting 必須要看 training set 的結果。而這張圖表，是 training set 的結果，所以這並不是 overfitting。這個是訓練時，就訓練失敗了。其中一個原因叫做 Vanishing Gradient。當你把 network 疊得很深的時候，在最靠近 input 的幾個層的這些參數，對最後 loss function 的微分值會很小；而在比較靠近 output 的幾個層的微分值會很大。因此，當你設定同樣的 learning rate 時，會發現靠近 input 的地方參數更新的速度是很慢的；靠近 output 的地方參數更新的速度是很快的。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/00cd9876-bd48-490e-ae7a-d6bf4a4ca831)

所以，你會發現在 input 參數幾乎還是 random 的時候，output 就已經收斂了。在靠近 input 地方的這些參數還是 random 值時，output 的地方就已經根據這些 random 的結果找到一個 local minimum，然後就收斂了。這時你會發現 loss 下降的速度變得很慢，就覺得這個 model 參數卡在 local minimum 之類的，就傷心地把程式停掉了。此時你得到的結果其實是很差的，為什麼呢？因為這個收斂的狀態幾乎基於 random 的參數，所以得到的結果其實是很差的。

為甚麼會有這個現象發生呢？如果 Backpropagation 的式子寫出來的話便可以很輕易地發現 sigmoid function 會導致這種情況發生。但是就算不看 Backpropagation 的式子，從直覺上來想也可以了解為什麼這種情況會發生。用直覺來想，一個參數的 Gradient 的值應該是某一個參數 w 對 total cost C 的偏微分。也就是說，它直覺上的意思就是當我把某一個參數做小小的變化時，它對這個 cost 的影響。我們可以把一個參數做小小的變化，然後觀察它對 cost 的變化，而藉此來決定這個參數的 Gradient 的值有多大。

所以我們就把第一個 layer 裡面的某一個參數加上 Delta w，看看對 network 的 output 和 target 之間的 loss 有甚麼樣的影響。你會發現，如果今天這個 Delta w 很大，在通過 sigmoid function 的時候是會變小的。也就是說，改變了某一個參數的 weight，對某一個 neuron 的 output 的值會有影響，但是這個影響會衰減。因為，假設用 sigmoid function，它的形狀長的如下圖：

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/23637c6e-76df-464a-97cb-df884e54272b)

sigmoid function 會把負無窮大到正無窮大之間的值都硬壓到 0~1 之間。也就是說，如果有很大的 input 的變化，在通過 sigmoid function 以後，它對 output 的變化會很小。所以，就算今天這個 Delta w 有很大的變化，造成 sigmoid function 的 input 有很大的變化，對 sigmoid function 來說，它的 output 的變化是會衰減的。而每通過一次 sigmoid function，變化就衰減一次。所以當 network 越深，它衰減的次數就越多，直到最後，它對output 的影響是非常小的。換句話說，你在 input 的地方改一下你的參數，對它最後 output 的變化，其實是很小的，因此最後對 cost 的影響也很小。這樣會導致靠近 input 的那些 weight，它對 Gradient 的影響是小的。

梯度消失（Vanishing Gradient）的原因，1.导数的范围：在深层网络中，每层的梯度都依赖于前一层的梯度。当你使用 Sigmoid 函数时，每通过一层，梯度就可能被缩小，最终可能变得非常小，以至于在网络的较低层几乎没有有效的梯度传递，这就是梯度消失问题。2.网络深度：
随着网络层的增加，连续乘以小于 1 的数会使梯度指数级减小。这意味着网络中更深的层在训练过程中几乎不会更新权重，从而难以学习。

虽然合理的初始化（比如 He 或 Xavier 初始化）可以在一定程度上帮助缓解梯度消失的问题，但它们并不能完全解决由 Sigmoid 激活函数引起的根本问题。良好的初始值可以帮助梯度在训练初期保持有效的范围，但随着训练的深入，梯度消失的问题可能仍会出现，特别是在非常深的网络中。

解决方案：

1.使用 ReLU 及其变体：ReLU 激活函数及其变体（如 Leaky ReLU、Parametric ReLU）在正区间内的梯度是常数（通常为1），这帮助避免了在正输入值下的梯度消失问题。

2.批量归一化（Batch Normalization）：这种方法可以帮助调节各层的输入，使其保持在激活函数的线性区域，从而有助于缓解梯度消失问题。

3.更加谨慎的网络设计：避免过深的网络，或者使用跳跃连接（如在残差网络中）来保持梯度流。

以下就介绍第一种方法 ReLU 
比較早年的做法是去訓練 RBM，去做 layer-wise 的 training。也就是說，先訓練好一個 layer。因為如果把所有的這個 network 兜起來，那在做 Backpropagation 的時候，第一個 layer 幾乎沒有辦法被訓練到。所以，RBM 的精神就是：先把一個 layer train 好之後，再 train 第二個，再 train 第三個。最後在做 Backpropagation 的時候，雖然第一個 layer 幾乎沒有被 train 到也所謂，因為一開始在 pre-train 的時候，就把它 pre-train 好了。以上就是 RBM pre-train 為什麼可能有用的原因。

後來 Hinton 跟 Pengel 都幾乎在同樣的時間，不約而同地提出同樣的想法：改一下 activation function，可能就可以解決這個問題了。所以，現在比較常用的 activation function，叫做 Rectified Linear Unit，它的縮寫是 ReLU。這個 activation function 的函數圖形如下圖：

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/dc3f26de-1daa-4850-b1f1-754f9d0e4bac)

其中 z 是 activation function 的 input，a 是 activation function 的 output。如果 activation function 的 input 大於 0，input 就會等於 output；如果 activation function 的 input 小於 0，output 就是 0。

選擇這樣的 activation function 有甚麼好處呢？有以下幾個理由：第一個理由是它比較快，跟 sigmoid function 比起來，它的運算是快很多的。sigmoid function 裡面的 exponential 運算是很慢的，使用這個方法快得多。Pengel 的原始論文有提到這個 activation function 的想法其實有一些生命上的理由，而他把這樣的 activation 跟一些生物上的觀察結合在一起。

而 Hinton 則說過像 ReLU 這樣的 activation function 其實等同於無窮多的 sigmoid function 疊加的結果。那些無窮多的 sigmoid function 之中，它們的 bias 都不一樣，而疊加的結果會變成 ReLU 的 activation function。但它最重要的理由是它可以處理 Vanishing gradient 的問題。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0b60fc38-547d-42ee-91ce-c993916a5df3)

我們可以觀察以上這個 ReLU 的 neural network。它裡面的每一個 activation function 都是 ReLU 的 activation function。ReLU 的 activation function 作用在兩個不同的 region：一個 region 是當 activation function 的 input 大於 0 時，input 會等於 output；另外一個 region 是 activation function 的 input 小於 0 時，output 就是 0。所以，現在每一個 ReLU 的 activation function，都作用在以上提到的兩個不同的 region。當 input = output 時，這個 activation function 其實就是 linear 的；那對那些 output 是 0 的 neuron 來說，它其實對整個 network 是一點影響都沒有的：因為它 output 是 0，所以它根本就不會影響最後 output 的值。假如有一個 neuron 它 output 是 0 的話，根本就可以把它從 network 裡面整個拿掉。當你把這些 output 是 0 的 network 拿掉，剩下的 neuron，就都是 input 等於 output，也就是 linear 時，整個 network 不就可以如下圖一樣，看成是一個很瘦長的linear network 嗎？

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/8cc19094-872a-4ae7-b2f7-ab3c4d4c0685)

我們剛才提到： Gradient 會遞減，是因為通過 sigmoid function 的關係，原因是 sigmoid function 會把比較大的 input 變成比較小的 output。但是，如果是 linear network 的話，input 等於 output，就不會有所謂 activation function 遞減的問題了。

這邊有些人可能會有一個問題：如果用 ReLU 的話，整個 network 會變成 linear network ，可是我們要的並不是一個 linear 的 network。我們之所以要用 deep learning，就是因為我們不想要我們的 function 是 linear 的，而是希望它是一個 non-linear、一個比較複雜的 function。當我們用 ReLU 的時候，它不就變成一個 linear 的 function 了嗎？這樣不是變得很弱嗎？

其實是因為這個 network 整體來說依舊是 non-linear 的。當每一個 neuron，它 operation 的 region 是一樣的時候，它是 linear 的，也就是說，如果對 input 做小小的改變，不去改變 neuron 的 operation 的 region 的話，它是一個 linear 的 function；但是如果對 input 做比較大的改變，改變了 neuron 的 operation region 的話，它就變成是 non-linear 的。

還有另外常被問到的問題：ReLU 不能微分。我們之前提到，在做 Gradient Descent 時，需要對 loss function 做微分。也就是說，neural network 要是一個可微的 function，但是 ReLU 不可微分，至少在原點是不可微的。那我們應該怎麼辦呢？

其實在實作上可以這樣解決：當 region 在原點右側的時候，gradient 就是 1；region 在原點左側的時候，微分就是 0。因為不可能會發生 input 正好是 0 的情況，就忽略它。

所以可以尝试将之前做好的Keras_helloworld_DNN.py将activation改成ReLu试试看，是有些微改善。那我们尝试根据以下几点做改善：

1.优化器和学习率调整
使用不同的优化器可能会显著影响训练结果。虽然已经使用了SGD，但可以考虑使用具有自适应学习率的优化器，如Adam，这通常能提供更好的收敛性能。

2. 更换损失函数
对于多分类问题，categorical_crossentropy 比 mse（均方误差）通常效果更好，因为它直接针对概率分布进行优化，更适合分类问题。

3. 减少模型复杂度
范例模型有很多层和神经元（633个单元和10层），这可能导致过拟合，尤其是当训练样本相对较少（只有10000个）时。尝试简化模型，比如减少层数或每层的单元数。

4. 增加Dropout层
为了减少过拟合，可以在几个全连接层之间加入Dropout层。Dropout层在训练过程中随机丢弃一部分网络连接，这有助于防止模型过度依赖训练数据中的任何单个节点。

5. 数据扩增
如果训练数据不足，可以通过数据扩增来人为地增加训练样本的多样性，这通常能帮助提高模型的泛化能力。对于图像数据，常见的扩增技术包括旋转、缩放、翻转等。

6. 批量归一化
批量归一化可以加速深度网络的训练，减少初始化对训练的影响，同时也有轻微的正则化效果。可以在每层或特定层后添加批量归一化。

7. 更早的停止训练
如果观察到训练准确率持续提高而验证准确率开始下降，可能是过拟合的标志。使用早停（Early Stopping）策略可以在验证损失不再改善时自动停止训练。

根据以上几点做改善，将code改写成Keras_optimization_helloworld_DNN.py，正确率就大大提升。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3e1ea7f4-9174-4651-860f-993d6909a715)

而 ReLU 其實還有各種變體。有人覺得原來的 ReLU 在 input 小於 0 的時候 output 會是 0，在這個時候微分是 0，就沒有辦法 update 參數了。所以，我們應該在 input 小於 0 的時候，output 還是有一點點的值，也就是說 input 小於 0 的時候，output 是 input 乘上 0.01，這個函數叫做 Leaky ReLU。

那此時，有人就會提出問題：為甚麼是 0.01，而不是 0.07, 0.08 之類的數值呢？所以，就有人提出了 Parametric ReLU：在負的這側呢 a = z * alpha 。alpha 是一個 network 的參數，它可以透過 training data 被學出來，甚至每一個 neuron 都可以有不同的 alpha 的值。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/463aba09-0609-4154-9755-77e70e33ba9b)

那又會有人問為甚麼一定要是 ReLU 這個樣子呢？所以，後來又有一個更進階的想法叫做 Maxout network。Maxout network 會讓 network 自動學它的 activation function。而因為現在 activation function 是自動學出來的，所以 ReLU 就只是 Maxout network 的一個 special case。也就是說 Maxout network 可以學出像 ReLU 這樣的 activation function，但是也可以學出其他的 activation function，training data 會決定現在的 activation function 應該要長甚麼樣子。

假設現在 input 是一個 2 dimension 的 vector [x_1, x_2] 。然後把 [x_1, x_2] 乘上不同的 weight

分別得到四個 value： 5, 7, -1, 1。本來這四個值應該要通過 activation function，不管是 sigmoid function 還是 ReLU，來得到另外一個 value。但是在 Maxout network 裡面我們會把這些 value group 起來，而哪些 value 應該被 group 起來這件事情是事先決定的。比如說，在這個例子中以上這兩個 value 是一組，以下這兩個 value 是一組，然後在同一個組裡面選一個值最大的當作 output。比如說，上面這個組就選 7，而下面這個組就選 1。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/548c8584-b213-4cdd-a280-c64edc6b6ab7)

這件事情其實就跟 Max Pooling 一樣，只是我們現在不是在 image 上做 Max Pooling，而是在一個 layer 上做 Max Pooling。我們把本來要放到 neuron 的 activation function 的這些 input 的值 group 起來，然後只選 max 當作 output，這樣就不用 activation function，得到的值是 7 跟 1。這個作法就是一個 neuron，只是它的 output 是一個 vector，而不是一個值。那接下來這兩個值乘上不同的 weight，就會得到另外一排不同的值，然後一樣把它們做 grouping。我們一樣從每個 group 裡面選最大的值：1 跟 2 就選 2，4 跟 3 就選 4。在實作上，幾個 element 要不要放在同一個 group 裡面，是你可以自己決定的。這就跟 network structure 一樣，是你自己需要調的參數。所以，你可以不是兩個 element 放一組，而是 3 個、4 個、5 個都可以。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f315cd80-71bc-4487-8fa0-4f4f65a2a57d)







