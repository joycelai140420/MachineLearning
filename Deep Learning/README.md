首先这篇很长，实在是台大老师讲的重点太多，等我有空在分门别类...


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

你可以参考Maxout network.py范例做法。

在Training 时期修正模型还有一个方法就是adaptive 的 learning rate。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c5e3c146-d2b0-42d0-9a66-ec9cb850cae6)

其實 adaptive 的 learning rate，之前已經有介紹過：之前提到過的 Adagrad，做法就是每一個 parameter 都要有不同的 learning rate。於是我們就把一個固定的 learning rate eta 除掉這一個參數過去所有 gradient 值的平方和開根號，就得到新的 parameter。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/bf34f74e-7847-4c48-a4e0-86325d520529)


而 Adagrad 的精神就是假設我們今天考慮兩個參數 w_1 和 w_2，w_1 是在水平的方向上，它平常 gradient 都比較小，那它是比較平坦的，給它比較大的 learning rate。反過來說，在垂直方向上平常 gradient 都是比較大的，所以它是比較陡峭的，給它比較小的 learning rate。但是實際上我們面對的問題有可能是比 Adagrad 可以處理的問題更加複雜的。也就是說，我們之前在做這個 Linear Regression 的時候，我們觀察到的 optimization 的 loss function 是像上圖的 convex 的形狀。但實際上，當我們在做 deep learning 的時候，這個 loss function 可以是任何形狀。

Adagrad 的主要思想是对频繁出现的特征降低其学习率，对不频繁的特征提高其学习率。这使得模型在学习不频繁特征时更为敏感，从而能更好地捕捉稀疏数据中的信息。

优点：

自适应学习率：各参数有各自的学习率，对于稀疏数据的特征能自动调整学习速度，有助于更快的学习这些特征。

简单易实现：与标准的梯度下降相比，Adagrad 只需要在更新步骤中加入梯度的历史信息。

缺点：

学习率单调递减：由于累积梯度平方和只增不减，导致学习率随着迭代逐渐减小并最终趋近于零，这可能使得训练过程在后期几乎停滞不前。

对初始学习率敏感：初始学习率的选择可能对模型的性能有较大影响。

由于这些缺点，Adagrad 在实践中常被其改进版本，如 AdaDelta 或 Adam 所取代，这些算法试图解决学习率持续减小的问题。你可以参考Maxout network.py里面已经使用Adam（model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
）

因为累积梯度平方和只增不减，导致学习率随着迭代逐渐减小并最终趋近于零，这可能使得训练过程在后期几乎停滞不前。所以有一個 Adagrad 的進階版叫做 RMSProp。

RMSProp（Root Mean Square Propagation）是由 Geoff Hinton 提出的一种自适应学习率方法，旨在解决 Adagrad 在训练深层神经网络时遇到的一个主要问题：学习率随着迭代不断减小至接近零，使得模型在训练后期几乎无法进一步优化。RMSProp 通过引入衰减系数来解决 Adagrad 学习率不断减小的问题。它保持一个移动（衰减的）平均值对梯度的平方，而不是像 Adagrad 那样累积所有过去梯度的平方。

算法步骤

1.选择一个初始学习率 η 和两个参数：衰减率γ（通常设为 0.9）和一个小常数ϵ（通常是1𝑒−8），用于保持数值稳定性，避免除零错误。

2.计算梯度 gt在每个参数上的偏导数。

3.更新累积的平方梯度 vt。与 Adagrad 直接累计所有历史梯度平方不同，RMSProp 计算梯度平方的指数加权移动平均：

![1714288380453](https://github.com/joycelai140420/MachineLearning/assets/167413809/ccc0b31a-a76c-4caa-b7f6-93c51f4e98cd)

4.调整每个参数。与 Adagrad 类似，参数更新使用计算出的梯度和调整后的学习率：
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3a643af1-97ae-4fa9-a2f6-1ea05c48bebb)

RMSProp 是否能改善 Adagrad 的问题？

RMSProp 的主要改进在于其使用移动平均而非累积所有历史梯度平方，这意味着它不会让学习率持续降低至接近零。这使得 RMSProp 在线和非静态设置中特别有用，因为它可以快速适应新的数据模式。此外，它常常能有效避免深层网络训练过程中的梯度消失问题。

你可以参考RMSProp.py范例来了解运作逻辑。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/40aa47d5-27ec-4c28-9a8e-2cb828156cf0)

除了 learning rate 的問題以外，我們知道在做 deep learning 的時候會卡在 local minimum，之前也有提過不見得是卡在 local minimum，也有可能卡在 saddle point，甚至是卡在 plateau 的地方。大家對這個問題都非常的擔心，覺得 deep learning 是非常困難的，因為可能胡亂做一下就產生很多問題。其實 Yann LeCun 在 2007 年的時候，提出一個滿特別的說法：不用擔心 local minimum 的問題。其實在 error surface 上沒有太多 local minimum，因為要是一個 local minimum，必須在每一個 dimension 都要是一個谷底的形狀。假設山谷的谷底出現的機率是 p ，因為 network 有非常非常多的參數，所以假設有 1000 個參數，每一個參數都要是山谷的谷底的機率就是 p^1000。network 越大、參數越多，出現的機率就越低。所以 local minimum 在一個很大的 neural network 裡面，其實沒有想像中的那麼多。一個很大的 neural network，它看起來其實搞不好是很平滑的，根本沒有太多 local minimum。所以當走到一個你覺得是 local minimum 的地方而卡住的時候，它八成就是 local minimum，或是很接近 local minimum。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b721d315-2f4a-44d7-a621-9c9bc8275a74)

有一個 heuristic 的方法，可以稍微處理上述說的 local minimum 還有 plateau 的問題，這個方法可以說是從真實的世界得到的一些靈感。我們知道在真實的世界裡面，把一個球從上圖的左上角讓它滾下來，在球滾到 plateau 的地方時，因為有慣性，所以它不會停下來，而是會繼續往前。就算是走到上坡的地方，假設這個坡沒有很陡，因為慣性的關係，可能還是可以翻過這個山坡，結果它就走到了比這個 local minimum 還要好的地方。所以我們就把這個慣性的特性加到 Gradient Descent 裡面去，這就叫做 momentum。

动量（Momentum）优化器简介

动量方法是模拟物理学中粒子在下坡时积累速度的概念，用于加速梯度下降算法，尤其是在面对高曲率、小且不一致的梯度或噪声很多的梯度时。它有助于加快学习过程，并且能够减少训练过程中的振荡。
动量方法不仅考虑当前步的梯度，还会累积过去梯度的“动量”，以此决定当前更新的方向，这通常使得优化过程更快收敛，并减少震荡。

算法步骤

1.引入动量变量v，初始值为 0。

2.计算梯度gt ，即在参数θ 上的损失函数的导数。

3.更新动量：
![1714297323156](https://github.com/joycelai140420/MachineLearning/assets/167413809/f822ad9b-105a-4cd9-ae76-294c408bae95)
其中μ（一般取值约为 0.9）是动量因子，决定了之前梯度的保留量；η 是学习率。

更新参数：
𝜃=𝜃+𝑣𝑡
参数更新不仅由当前梯度决定，还受之前梯度的影响。
​
动量方法尤其适用于处理：
非凸优化问题：可以帮助跳出局部最小值。
深层网络的训练：有助于在深层网络和复杂表面中快速前进。
面对小的/不一致的梯度问题：动量可以推动过程跨过平坦区域。

那么momentum 设置要怎么设定大小呢？

动量系数通常用符号 𝜇表示，并且是一个介于 0 和 1 之间的值。设置为 0.9 是一种常见的选择，但理解何时应该调整这个值更大或更小是优化训练过程中的关键。
在大多数情况下，设置为 0.9 提供了一个良好的平衡，能够有效地加速学习过程并避免过度振荡。这个值允许算法保持大部分之前的梯度方向（90%），从而在参数更新时累积较长的“动量”，推动参数沿着持续的方向更新，这有助于快速穿越平坦区域并避免被小的局部极值点捕获。

适用场景：特别适用于深度网络或复杂的误差表面，在这些问题中常常需要有效地从梯度的噪声中恢复出有用的信息，并且需要稳定但又不是过于缓慢的更新。

较高的动量值（接近 1）：

优点：可以更快地收敛，尤其是在目标函数的形状比较复杂，或者梯度具有高度非一致性的情况下。

风险：较高的动量值可能会导致优化过程在遇到局部最小或鞍点时越过最优解，因为累积的动量可能过大，使得参数更新步伐过大。

适用场景：在梯度变化不大且稳定的大型数据集上效果较好。

较低的动量值（接近 0）：

优点：更新更加谨慎，能够更细致地探索误差表面，减少对过往梯度信息的依赖，有助于精细调整解。

风险：收敛速度可能较慢，特别是在误差表面平坦的区域，缺乏足够的动量可能使得学习过程陷入停滞。

适用场景：对于小型数据集或者模型较简单时，较低的动量可以帮助模型避免过快收敛到非最优解。

在实践中，选择动量的具体值通常需要基于经验和实验。在开始时使用标准值（如 0.9）是一个合理的起点。随后，根据模型在验证集上的表现进行调整：如果模型训练过程中表现出振荡或更新过快，可以尝试降低动量值；如果训练过程缓慢，可以尝试增加动量值。此外，也可以通过交叉验证等技术来找到最优的动量设置。

你可以参考Momentum.py范例来了解运作逻辑。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/6c127be7-1303-4851-b3ab-ad3edabf7cc5)

我們剛才所討論的都是如果在 training data 上的結果不好的話怎麼辦。接下來要討論的則是如果今天已經在 training data 上得到夠好的結果，但是在 testing data 上的結果仍然不好，那有甚麼可行的方法。接下來會很快介紹 3 個方法：Early Stopping、Regularization 跟 Dropout。Early Stopping 跟 Regularization 是很 typical 的作法，他們不是 specific design for deep learning 的，是一個很傳統、typical 的作法。而 Dropout 是一個滿有 deep learning 特色的做法，在介紹 deep learning 的時候，需要討論一下。在前面几个py范例都有引用Early Stopping做法来提供参考。发现在添加Early Stopping也可以避免等太久。如果观察到训练准确率持续提高而验证准确率开始下降，可能是过拟合的标志。使用早停（Early Stopping）策略可以在验证损失不再改善时自动停止训练。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0d04f816-eadf-49eb-bc8b-6a88bcb017df)
其实Adam 就是RMSProp+Momentum

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a9c474c6-f3d7-460e-bfc2-e894c5ba85f9)

首先我們來介紹一下 Early Stopping，Early Stopping 是甚麼意思呢？我們知道，隨著你的 training，如果learning rate 調的對的話，total loss 通常會越來越小；那如果 rate 沒有設好，loss 變大也是有可能的。想像一下如果 learning rate 調的很好的話，那在 training set 上的 loss 應該是逐漸變小的。但是因為 training set 跟 testing set 他們的 distribution 並不完全一樣，所以有可能當 training 的 loss 逐漸減小的時候，testing data 的 loss 卻反而上升了。

所以理想上，假如知道 testing data 的 loss 的變化，我們應該停在不是 training set 的 loss 最小、而是 testing set 的 loss 最小的地方。在 train 的時候，不要一直 train 下去，可能 train 到中間這個地方的時候，就要停下來了。但是實際上，我們不知道 testing set，根本不知道 testing set 的 error 是甚麼。所以我們其實會用 validation set 來 verify 這件事情。這邊的 testing set，並不是指真正的 testing set。它指的是有 label data 的 testing set。比如說，如果你今天是在做作業的時候這邊的 testing set 可能指的是 Kaggle 上的 public set；或者是，你自己切出來的 validation set。但是我們不會知道真正的 testing set 的變化所以其實我們會用 validation set 模擬 testing set 來看甚麼時候是 validation set 的 loss 最小的時候，並且把 training 停下來。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/55fee3b1-b822-4d3d-9781-295f63fc7133)

那 Regularization 是甚麼呢？我們重新定義了那個我們要去 minimize 的 loss function。我們原來要 minimize 的 loss function 是 define 在你的 training data 上的，比如說要 minimize square error 或 minimize cross entropy。那在做 Regularization 的時候我們會加另外一個 Regularization 的 term，比如說，它可以是你的參數的 L2 norm。假設現在我們的參數 theta 裡面，它是一群參數，w_1, w_2 等等有一大堆的參數。那這個 theta 的 L2 norm 就是 model 裡面的每一個參數都取平方然後加起來，也就是這個 lVert theta rVert_2 。因為現在用 L2 norm 來做 Regularization，所以我們稱之為 L2 的 Regularization。

我們之前有提過，在做 Regularization 的時候一般是不會考慮 bias 這項。因為加 Regularization 的目的是為了要讓我們的 function 更平滑，而 bias 通常跟 function 的平滑程度是沒有關係的。所以，通常在算 Regularization 的時候不會把 bias 考慮進來。

Regularization是机器学习中用于减少模型过拟合的一种技术。过拟合是指模型在训练数据上表现出色，但在新的、未见过的数据上表现较差的现象。正则化通过添加额外的信息（通常是一种惩罚项）到损失函数中来抑制过度复杂的模型。

为什么使用正则化可以改善模型表现？

控制模型复杂度：正则化通过惩罚模型的复杂度（如权重的大小）来限制模型的自由度，这有助于避免过度拟合训练数据的详细噪声。

提高泛化能力：通过防止模型过度依赖训练数据中的特定特征，正则化有助于提升模型在未见数据上的表现。

避免学习过于极端的模型权重：在许多情况下，具有较小权重的模型更简单，对输入数据中的小变化不那么敏感，因此更稳定。

L1正则化（Lasso Regularization）：
将权重的绝对值的总和加到原来的损失函数上。因为是每一次减到固定值，所以通常遇到很大W时候也有可能训练出来的模型也有很大loss。

优点：可以产生稀疏权重（许多权重为0），从而实现特征选择。

缺点：可能不稳定，即小的数据变化可能导致生成的模型差异较大。

L2正则化（Ridge Regularization）：
将权重的平方和加到原来的损失函数上。因为是每一次乘上一个小于1的固定值，所以遇到很大W时候也会被压出小的loss。

优点：通常会使学习过程更加平滑，减少数据随机性带来的影响。

缺点：不会产生稀疏解，因为权重很少真正变成零。

何时使用正则化最好？

面对过拟合时：当训练误差远小于测试误差时，通常意味着模型过拟合。

数据集较小：小数据集更容易过拟合，使用正则化可以帮助模型提高泛化性能。

模型过于复杂：对于具有大量参数的复杂模型，尤其是深度神经网络，正则化是必需的。

至于在设置 L2 正则化的系数时，选择合适的数值是一个需要仔细考虑的问题，因为它会直接影响模型的训练过程和最终性能。

使用验证集调整

使用一个独立的验证数据集来评估不同正则化系数的效果。可以选定一个系数的范围，并通过交叉验证来测试每个值的效果。

较小的系数：如果正则化系数太小，可能无法有效防止过拟合。如果你看到训练误差远低于验证误差，可能需要增加正则化系数。

较大的系数：如果正则化系数太大，可能导致模型欠拟合，即模型在训练数据上的表现也不佳。如果训练误差和验证误差都很高，应考虑减小正则化系数。

观察学习曲线

观察不同正则化值下的学习曲线（训练和验证误差）是非常有帮助的。理想的学习曲线应该显示训练误差和验证误差逐渐接近，并且两者都逐渐减小。

在Keras中实现L2正则化非常简单，可以直接在层中添加正则化器：
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))

但有见解是Regularization效果在DNN，没有比较好，不如Early Stopping，两者的目的都是为了参数不要离0太远，Early Stopping会比较好用。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/17037abb-a84a-429f-9b48-0807b77c7e8e)

最後我們要介紹一下 dropout。我們先介紹 dropout 是怎麼做的，然後才說明為甚麼這樣做。dropout 是怎麼做的呢？它是這樣，在 training 的時候，每一次我們要 update 參數之前，我們都對每一個 neuron，包括 input layer 裡面的每一個 element 做 sampling，那這個 sampling 是決定這個 neuron 要不要被丟掉，每個 neuron 有 p 的機率會被丟掉。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/87c94f79-d189-4471-8aff-d42d84cf8665)

那如果一個 neuron 被 sample 到要丟掉的時候，跟它相連的 weight也失去作用，所以就變上圖這樣。所以，做完這個 sample 以後，network 的 structure 就變瘦了，變得比較細長，然後再去 train 這個比較細長的 network。這邊要注意一下，所謂的 sampling 是每次 update 參數之前都要做一次，每一次 update 參數的時候 training 的那個 network structure 是不一樣的。當你在 training 使用 dropout 的時候，performance 會變差，因為本來如果你不要 dropout 好好的做的話，在 MNIST 上，可以把正確率做到 100%。但是如果加 dropout，因為神經元在 train 時有時候莫名其妙就會不見，所以你在 training 時有時候 performance 會變差，本來可以 train 到 100%，就會變成只剩下 98%。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f1ecf3df-849a-47dd-b2f5-5ea1dc987ede)

所以，當你加了 dropout 的時候，在 training 上會看到結果變差。dropout 它真正要做的事情是就是要讓 training 的結果變差，但是 testing 的結果是變好的。也就是說，如果你今天遇到的問題是 training 做得不夠好，你再加 dropout，就是越做越差那在 testing 的時候怎麼做呢？在 testing 的時候要注意兩件事：第一件事情就是 testing 的時候所有的 neuron 都要用，不做 dropout；另外一件事情是，假設在 training 時，dropout rate 是 p，那在 testing 的時候，所有 weight 都要乘 (1 - p%)。也就是說，假設現在 dropout rate 是 50%，在 training 時 learn 出來的 weight 等於 1，那 testing 的時候，你要把那個 weight 設成 0.5。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d6f59f26-d136-4dea-9eef-18c1ea4d5eee)

另外想解釋的就是為甚麼 dropout rate 50% 的時候，testing 的 weight 就要乘 0.5？為甚麼 training 跟 testing 的 weight 是不一樣的呢？因為照理說 training 用甚麼 weight 就要用在 testing 上，training 跟 testing 的時候居然是用不同的 weight，為甚麼這樣呢？直覺的理由是：假設現在 dropout rate 是 50%，那在 training 的時候期望總是會丟掉一半的 neuron。對每一個 neuron 來說，總是期望說它有一半的 neuron 是不見的，是沒有 input 的。所以假設在這個情況下，learn 好一組 weight，但是在 testing 的時候，是沒有 dropout 的。對同一組 weight 來說，假如你在這邊用這組 weight 得到 z，跟在這邊用這組 weight 得到 z'，它們得到的值，其實是會差兩倍。因為在左邊這個情況下，總是會有一半的 input 不見；在右邊這個情況下，所有的 input 都會在。而用同一組 weight 的話，變成 z'就是 z 的兩倍了，這樣變成 training 跟 testing 不 match，performance 反而會變差。所以把所有 weight 都乘 0.5，做一下 normalization，這樣 z 就會等於 z'。把這個 weight 乘上一個值以後，反而會讓 training 跟 testing 是比較 match 的。這個是比較直觀上的結果，如果要更正式的話，其實 dropout 有很多理由。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c2756978-3975-446e-b862-d8fbc64146e1)

![Uploading image.png…]()


前面Keras_optimization_helloworld_DNN.py有使用到dropout可以参考。
