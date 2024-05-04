看着前面台大老师介绍ecurrent Neural Network (Part I)，我们在跟随着老师课程在进一步学习ecurrent Neural Network 的知识。



ecurrent Neural Network (Part II) 

如何學習 How to Learning
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a1f23981-6cc3-4505-bc61-ab65b40c17b3)

須先定義 Cost function 以及 Loss

Example: Slot Filling

    每一個output有其對應的slot

        第一個 word 屬於 other 這個 slot
        台北 屬於 destination 這個 slot
        on 屬於 other slot
        November 2nd 屬於抵達時間的 slot
        
    算 每一個時間點 的 RNN output 跟 reference vector 的 cross entropy

        slot 的 cross entropy
        
    得到 loss function，接著 gradient decent

    Back propagation



Back propagation Through Time
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/bcd171a2-548f-4cfe-87a5-602c20603cf6)

BPTT，Back propagation 的進階版

因為 RNN 是在 time sequence 上運作，所以 BPTT 需要考慮時間的information

不詳細說明，只需要知道 RNN 用 gradient decent train 可以 train


困難點
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7b9881d7-d10b-4011-b5b1-5a3047d04dc2)

有時訓練無法像藍色一樣順利，會向綠色一樣total loss，不是這沒幸運像藍色一樣。

這個 learning curve ，抖到某個地方，就突然 NaN，然後程式就segmentation fault




RNN 的 Error Surface
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2bdd41b8-67c3-4b56-9bf9-c2f5cf5aeb3c)

造成綠色線的分析原因：RNN 的 error surface 非常的陡峭

    error surface 就是 Total Loss 對參數的變化
    
    崎嶇的意思是：這個 error surface 有些地方非常平坦，有些地方非常陡峭
    
舉例，參考上圖：

    假設從橙色那個點開始，跳到下一個橙色的點

    可能正好就跳過一個懸崖 => Loss 會突然暴增

        因為之前 gradient 很小，所以 learning rate 調得比較大
        gradient 突然很大 => 很大的 gradient 再乘上很大的 learning rate（就飛出去）
        參數就 update 很多，就 NaN => 程式就 segmentation fault
        
解決辦法

    clipping

        當 gradient 大於某一個threshold 時後，不要讓他超過那個 threshold
        Ex：當 gradient 大於 15 的時候就等於15（如何）


这种常见的“梯度爆炸”问题。模型的权重更新过大，导致训练过程中损失函数（loss）迅速增大或变得非常不稳定，这通常会使得模型完全无法收敛。“梯度爆炸”问题，特别是在权重初始化不当或训练数据特别多变时更为常见。

为了防止梯度爆炸，一种常用的技术是梯度裁剪梯度裁剪（Gradient Clipping），设置梯度裁剪的阈值并没有固定的规则，它通常需要根据具体的应用和数据进行实验调整。一般来说，阈值的选择可以从较小的值开始尝试，并通过实验来调整，以找到最佳的训练效果。常见的阈值范围可能在1到10之间。你可以开始于一个中等的值（比如5），然后根据模型在验证集上的表现来调整。你通常可以慢慢调整去观察现象。我们参考Gradient Clipping.py，来观察随着每一次的Training Steps进行Gradient Clipping后的损失值波动会慢慢变得没这么大下图，但还是要考虑其他技术，例如：调整模型复杂度、使用正则化、优化学习率和其他超参数、改进数据预处理，确保数据质量等等才成更提高模型的准确性。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0a5a6b9f-a77e-4c20-839b-3f2e6a83e42f)


為什麼 RNN 會有這種奇特的特性

sigmoid function 会造成gradient vanishing (梯度消失)，在之前有讲到ReLU提过，这边在进行粗略的说明。

sigmoid函数会导致梯度消失（gradient vanishing）的问题，主要是因为它在输入值非常大或非常小的时候，其导数（即梯度）会接近于零。sigmoid函数的输出范围在0到1之间。如果网络的输入值很大或很小，sigmoid函数会将这些值“压缩”到接近0或1的输出。这意味着，即使输入值有很大的变化，输出的变化也非常小。因此，当我们计算梯度时，即计算这个函数关于输入的导数，我们会得到一个非常小的值。在深度神经网络中，我们通过反向传播算法来更新网络的权重。这个过程涉及到多个这样的导数的连乘。如果每一层的梯度都很小，那么这些小的梯度相乘后，结果会越来越小，最终可能接近于零。这就是所谓的梯度消失问题，它会导致网络中较早层的权重几乎不更新，从而影响整个网络的学习效果。一般会用ReLU替代sigmoid函数去解决gradient vanishing 。

但是在RNN换成ReLU通常performance会比较差，所以activation funaction在RNN这里并不是这地方的关键点。

不过,在后面有Hinton论文问题到，用一般的RNN（非LSTM），用identity matrix来initialize transition的weight，然后再使用ReLU这个activation funaction时候，他可以得到很好的performance。如果适用一般的方法，initialize transition的weight是random的话，那sigmoid会比ReLU 的performance好。所以RNN用了 identity matrix + Relu的时候，performance有很大可能比LSTM好。

ps:单位矩阵（identity matrix），也称为恒等矩阵，是一种特殊的方阵。在这种矩阵中，主对角线（从左上角到右下角）上的元素都是1，而其余位置的元素都是0。
例如:
![1714791408771(1)](https://github.com/joycelai140420/MachineLearning/assets/167413809/b03c51e6-3823-4600-ae40-125793c63305)


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d88d4530-6565-4647-bf43-1b6dd51ef09d)

用直觀方法來知道一個 gradient 的大小

    把某一個參數做小小的變化，看他對 network output 的變化有多大
假設：

    一個簡單的 RNN，一個 linear neuron，weight 是 1，沒有 bias

    一個 input ，output weight 也是 1，transition 部分的 weight 是 w

    如果 input 是 [1 0 ... 0 0] => 第 1000 個時間點的 output 值是 w 的 999 次方

觀察：

    假設 w = 1，network 在最後時間點的 output 也是 1
    
    假設 w = 1.01，network 在最後時間點的 output 是 20000
    
    假設 w = 0.99，network 在最後時間點的 output 是 0

在 1 這個地方有很大的 gradient ，但在 0.99 的地方 gradient 就突然變得非常非常的小

    有時候需要一個很大的 learning rate
    
    設 learning rate 很麻煩，error surface 很崎嶇
    
RNN training 的問題

    來自於 RNN 把同樣的東西，在 transition 的時候反覆使用

    会造成 gradient vanishing 以及 gradient explode（观察上面）w = 1、w = 1.01、w = 0.99，经过w 的 999 次方后，就会造成gradient vanishing 以及 gradient explode。且不知道怎么设置Learning rate。

RNN training 的問題其实是来自于，他把同样的东西，在transtiion的时候，在时间和时间转换时候，反复使用，从memory 接到Neuron的那一组weight，在不同时间点，都是反复被使用到导致这样的场景。

解決方法
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/217013da-e95d-49f2-8fb8-b22383d969ed)

LSTM 可以讓 error surface 不要那麼崎嶇

    解決 gradient vanishing

    不會解決 gradient explode

關於 LSTM 的常見問答

    為什麼我們把 RNN 換成 LSTM ?

        因為 LSTM 可以 handle gradient vanishing 的問題
        
    為什麼 LSTM 可以 handle gradient vanishing 的問題呢？

        因為它們在面對 memory 的時候，處理的 operation 不一樣

            RNN 在每一個時間點 memory 裡面的資訊，都會被洗掉

            LSTM 會透過 forget gate 決定要不要洗去memory，如果過去的memory被影響，那這個影響有高機率會留著

        註：早期的LSTM是為了解決 gradient vanishing 的問題，一開始沒有 forget gate，後來才加上去的

Gated Recurrent Unit (GRU)

    用 gate 操控 memory 的 cell

    GRU 的 gate 只有 2 個

        參數量是比較少的，training時間少，比較robust
        
    精神：舊的不去，新的不來

        input gate 跟 forget gate 連動起來

        input gate 被打開的時候，forget gate 就會被自動關閉，反之亦然


更多處理 gradient vanishing 的 techniques

Clockwise RNN

SCRN

Vanilla(一般) RNN + identity matrix + ReLU activation function

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/59da32da-80b1-477e-bb9e-18fdb18ea21a)


更多應用

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/02304809-3901-4ad1-a93f-6e3b625f07a2)


RNN 可以做到更複雜的事情

    Sentiment Analysis

        用RNN自动learn一个classify，去分类那些文章是 positive 還是 negative
        
        例如：影評分析

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/89a6e183-1ed9-4690-ace0-1b59d30bfaba)

key term extraction

    Given 一篇文章， predict 這篇文章有那些關鍵詞彙

可以是RNN的最后一个时间的output做attention，把重要informaction 抽出来，在丢到feed forward network得到最后的output。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1deae325-e492-4e4e-a92a-b502c150863c)


也可以是多對多的
    語音辨識 (Speech Recognition)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/cfa4e62b-4376-48af-93ff-81e57fa15c0f)


input 是一串 acoustic feature sequence (每一小段時間切一個vector，0.01秒之類)

output 是 character 的 sequence

当input sequence长，output sequence短，例如语音辨识，一段语音翻译成一段文字。

Trimming

    好好好棒棒棒棒棒 => 好棒

    沒有辦法辨識 好棒棒 (與 好棒 意思相反)

CTC（解决，好棒（正向词汇），好棒棒（负向词汇）问题)
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c5623f2e-4a07-4432-a71e-f6d82a2fced2)


在 output 的時候，不只是 output 所有中文的 character。還多 output 一個符號，叫做 Null，叫做 沒有任何東西。

    output 就會變成是 好 null null 棒 null null null null 或是 好 null null 棒 null 棒 null null，能夠區分 好棒 跟 好棒棒

    解決疊字的問題



============================================================================

Sequence to sequence learning
    
    input 和 output 都是 sequence，但長度不一樣

        與CTC不同（input長，output 短），不確定誰長誰短



![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/bad82044-2fea-4921-a31a-39eb451be80d)

    實際case

        machine translation
        
            用 RNN 讀過去，在最後一個時間點呢，memory 存了所有 input 的整個 sequence 的 information（例如：讀進machine learning

            接下來，你就讓 RNN 吐開始吐 character（從機、器、學、習、慣、性⋯⋯

        問題：output停不下來

        解法：加一個「斷」的token


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/456fdf95-908b-4d93-835d-d988ca5464b6)

        同樣的做法可以用在語音辨識

            input：acoustic feature sequence

            output：character sequence

            沒有CTC強，但是意外的work

        結合語音辨識跟翻譯，例如：input是一段英文訊號，output中文句子

            搜集訓練資料變得比較容易（不需要中間的英文句子）

            

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/63b8080f-ed95-442a-93ea-f9f035766bac)

==========================================================================

Beyond Sequence
    Syntactic parsing tree：你输入一个句子，输出一个句子或语言结构的层次和组成部分之间关系的一种树状图。
    
        Input：一個句子

        Output：這個句子的文法的結構樹

    讓 machine 得到這樣的樹狀的結構

        過去：structure learning

        現在：把這個樹狀圖，描述成一個 sequence

            root 的地方是 S

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/817053dc-e3f3-432c-acdb-7a215a373bc8)


Document to vector

    bag-of-word

        忽略 word order 的 information，因为可能字的前后排序，导致意思都会差很多

        因为output时，字的排列可能造成句子正命或是負面的影響

    Sequence to sequence auto-encoder（比较能表达输入的文法意思）

        input 一個 word sequence

        通過 RNN 把它變成一個 embedded 的 vector（潛藏重要資訊）

        把這個 embedded vector 當成 decoder 的輸入，讓這個 decoder 長回一個一模一樣的句子

        好處：

            不需要 label data，只需要收集到大量的文章

        如果要得到语义的意思，用skip-thought，这个是输入一个句子，输出预测下一个句子
        
    Hierarchical neural auto-encoder

        output target會是下一個句子

        比較好得到 語意 的意思


那如果我們要把一個 document 表示成一個 vector 的話

往往會用 bag-of-word 的方法

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9c4f3bf3-4be8-42b3-ab9c-06216577a384)

語音上的應用

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/337f2281-df1f-4292-9d90-3352ae3f33e0)

audio 的 word to vector

    應用：語音搜尋

    使用者輸入一段話（語音），轉成vector後可以跟database裡的資料對比相似度。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0446a65e-0fb3-4424-ae03-b5b0372cda46)

先把audio segment 抽成 acoustic feature sequence當作input，丟入RNN（encoder）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e59640b2-38c3-47e5-82db-c107f63a5bfb)

然後再透過另外一個RNN decode，output要跟input越像越好

有趣的結果：

    各個字詞具有「聲音」上的意義，但沒有「語意」上的意義（把f換成n的，向量變化方向差不多）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/dbc7be03-ef37-4c67-ba8e-6df85fa767ac)

實際應用Demo
    
    Chatbot (聊天機器人)

        Sequence to sequence auto encoder

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/919c889b-6f45-4aea-8ca6-9a4da169fcdf)

    Attention-base model

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e46d6d5c-a8e9-4860-a20a-1576d6d7c8d9)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e87f4349-a75d-4d5d-b12e-7b0be5f27b79)


    當你輸入一個 input 的時候，這個 input 會被丟進一個中央處理器

    這個中央處理器，可能是一個 DNN/RNN，用來操控一個讀寫頭（reading head controller）

    這個 reading head controller 會決定這個 reading head放的位置，然後 machine 再從這個 reading head 放的位置，去讀取 information 出來，產生最後的 output

Neural Turing Machine（2.0 版本）

    會多去操控一個 writing head controller

    不只有讀的功能，還可以把資訊 discover 出來的東西，寫到它的 memory 裡面去


Reading comprehension
    attention-based model

    先讓model看一堆document，再問問題，透過attention model找出答案

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3c1f099d-a74c-43ce-8412-143f5c507ff7)


Visual Question Answering

    讓 machine 看一張圖，然後問它一個問題

    透過 CNN 可以把這個圖的每一小塊 region 用一個 vector 來表示
    
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/32bee25d-8bfe-44b8-98d4-34a6a6eaf61d)

Speech Question Answering

    讓 machine 聽一段聲音，然後問他問題，讓他從四個選項裡面選出正確選項（TOFEL）

    同樣使用 attention-based model

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4cd80d04-25ce-4714-9aaf-431f0c1eaa97)

RNN v.s. Structured learning
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d6f18856-7727-407e-81a3-985355249be4)

我這邊其實有一個問題

我們講了 Deep learning 也講了 Structured learning

    不同之處：

        考慮整個句子：

            uni-directional 的 RNN 或 LSTM只看一半

            透過 Viterbi 的 algorithm，可以考慮的是整個句子

            但RNN/LSTM 等等，也可以做 Bi-directional

        能否直接限制 label 和 label 之間的關係

            用 Viterbi algorithm 求解的時候，可以直接把你要的 constrain 下到 Viterbi algorithm 裡面

        RNN 和 LSTM 的 cost function跟實際上最後要考慮的 error 往往是沒有關係的

            如果是用 structured learning 的話，它的 cost會是你 error 的 upper bound

        Deep

            RNN/LSTM 可以是 deep

            HMM, CRF 拿來做 deep learning 是比較困難的

        整體說起來 RNN/LSTM 在 sequence labeling task 上面表現是比較好的

Deep learning 和 structured learning 的結合

    舉例：

        input 的 feature 先通過 RNN/LSTM

        通過 RNN/LSTM 的 output 再做為 HMM, CRF... 的 input

        用 RNN/LSTM 的 output 來定義 HMM, CRF... 的 evaluation function

    同時又享有 deep 的好處，又享有 structured learning 的好處

    語音上常見的組合

        deep learning 的 model: CNN/LSTM/DNN ，加上 HMM

        HMM 往往都還在

    Slot filling

        很流行用 Bi-directional LSTM，再加上 CRF 或是 structured SVM

        先用 Bi-directional LSTM 抽出 feature，再拿這些 feature 來定義 CRF 或者是 structured SVM 裡面需要用到的 feature

    也許 deep and structured 是未來一個研究的重點的方向


    







