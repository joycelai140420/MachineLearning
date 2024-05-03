
Recurrent Neural Network，簡稱RNN

是一種專門為處理序列數據（如時間序列或自然語言）設計的神經網絡。與傳統的神經網絡不同，RNN能夠在其隱藏層中保持一個內部狀態，這使得它能夠處理和記住輸入數據中的時間動態信息。RNN的原理是通過網絡的循環連接來實現的。在每個時間步驟，RNN都會接收到當前的輸入以及前一時間步的隱藏狀態。這個隱藏狀態可以被看作是網絡的“記憶”，它包含了過去信息的摘要，並用於當前的決策過程。RNN的一個關鍵特性是它們的權重在所有時間步中是共享的，這意味著學習到的模式可以應用於不同長度的序列。

RNN的優點包括：

    序列數據處理能力：能夠處理和生成基於時間序列的數據，如語音識別、語言模型等。
    參數共享：由於在所有時間步中權重是共享的，因此RNN可以用較少的參數處理較長的序列。

然而，RNN也有一些缺點：

    梯度消失和爆炸問題：在長序列中，RNN可能難以學習到依賴於遠距離的關係，因為梯度可能會在反向傳播過程中消失或爆炸。
    計算效率：由於序列的每個元素都需要前一步的輸出，這可能導致訓練過程中的計算效率降低。

為了解決這些問題，研究人員發展了RNN的變體，如長短期記憶網絡（Long Short-Term Memory，LSTM）和門控循環單元（Gated Recurrent Unit，GRU）。這些變體通過引入門機制來控制信息的流動，從而改善了RNN在處理長序列時的性能。


RNN的應用非常廣泛，包括：

    自然語言處理：如機器翻譯、情感分析、語言模型等。
    語音識別：將語音轉換為文字。
    時間序列預測：如股票價格預測、天氣預測等。

接下来是台大老师Hung-yi Lee授课内容

前導知識：字轉成vetctor的方式

1-of-N encoding（在前面的范例已经都介绍）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4b62a995-77c7-4bdc-b48c-5e04a95fc16a)

Beyond 1-of-N encoding

  1-of-N encoding 的一些問題

    起因：很多詞彙可能從來沒有見過
    解法：多加一個 dimension，這個 dimension 代表 other
    
  Beyond

    用某一個詞彙的字母來表示，像是 n-gram
    舉例 apple (3-gram)：有 "app"、 "ppl"、 "ple"

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f52265d9-c991-479c-9ded-ce211a96751e)

Task example: Slot Filling

  實際例子：智慧客服、訂票系統
  
  分析一段句子有哪些slot，如：Destination、Time of Arrival

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ef85a364-5b6d-4aaf-bed9-cae6f2003479)


解法 (Naive Model)

  可以用一個 Feedforward 的 Neural Network 來解

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e6e77e43-c004-41b2-bdf9-06c20479bc03)


  input 是一個詞彙（像 Taipei 可以轉成一個 vector）

  output 是一個 probability distribution，代表說我們現在 input 的這個詞彙屬於哪一個 slot 的機率 (像Taipei 屬於 destination 的機率，還是屬於 time of departure 的機率)

  Problem: example

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4563a269-3cfe-4207-9f1f-7df0a8556a33)

  Taipei 在下面的例子並不是destination，但對於model來說，上下input一樣，會得到相同結果。
  
  希望 model 可以有 記憶力（先看過arrive或是leave）

解法 (RNN)

Recurrent Neural Network，有記憶力的 neural network

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/95a0c409-ced8-48d7-8dc3-872bdf902140)

  每一次我們的 hidden layer 裡面的 neuron 產生 output 的時候，這個 output 都會被存到 memory（藍色的方塊）裡面去
  
  下一次當有 input 的時候，這些 neuron 不是只會考慮input 的這個 x1 跟 x2，它還會考慮存在 這些 memory 裡面的值（ hidden layer的值）
  
  換句話說：除了 x1 跟 x2 以外，存在 memory 裡面的值（a1、a2） 也會影響它的 output


實際例子

  假設條件：下圖的這個 network

    所有的 weight 都是 1
    所有的 neuron 沒有 bias 值
    所有的 activation function 都是 linear
    memory 起始值 0
  input 是 [1 1]、[1 1]、[2 2]

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/16009215-b7d1-4756-b1ed-0a56a6710829)

RNN吃到第一個 [1 1]

  第一層（綠色） Neuron 吃到的

    input（黃色）：[1 1]
    memory（藍色）：[0 0]
    output：2 （得到的 output 存回 memory）
    
第二層（紅色） Neuron 吃到的

    input（綠色）：[2 2]
    output：4
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/decfd747-a495-4c76-bfbc-4c2ff57eacff)

NN吃到第二個 [1 1]

  第一層（綠色） Neuron 吃到的

    input（黃色）：[1 1]
    memory（藍色）：[2 2]
    output： 2 + 2 + 1 + 1 = 6 （得到的 output 存回 memory）
    
  第二層（紅色） Neuron 吃到的
    
    input（綠色）：[6 6]
    output：12

小結論：

    對 Recurrent Neural Network 來說，就算是輸入一樣的東西，它的 output 是有可能會不一樣的
    因為存在 memory，裡面的值會改變結果

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/93bab400-313c-4d1c-b816-4db63bdfe8f8)

RNN吃到第三個 [2 2]

  memory 起始值 0

  第一層（綠色） Neuron 吃到的

    input（黃色）：[1 1]
    memory（藍色）：[6 6]
    output： 6 + 6 + 2 + 2 = 16 （得到的 output 存回 memory）

  第二層（紅色） Neuron 吃到的

    input（綠色）：[16 16]
    output：32

使用 RNN 要注意的事

  input 的 sequence 並不是 independent， sequence 的 order 很重要
  
  任意調換 input sequence 的順序，那 output 會完全不一樣

回歸 slot filling

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/48d079a9-3f66-4652-8fd8-7d0851936f1c)

有一個使用者說 arrive Taipei on November 2nd

當RNN 吃到 arrive

    那 arrive 就變成一個 vector，丟到 neural network裡面
    output 為 a1，是一個 vector
    根據 a1，產生 y1（arrive 屬於哪一個 slot 的機率）
    a1 會被存到 memory 裡面
當RNN 吃到 Taipei

    hidden layer 會同時考慮 Taipei 這個 input 跟存在 memory 裡面的 a1，得到 a2
    根據 a2，產生 y2（Taipei 屬於哪一個 slot 的機率）

依此類推 ...

並非3個 network，而是同一個 network在 三個不同的時間點，被 使用了3次
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a0c5594f-13c3-4317-bd35-78acf44eb1e7)

所以有了 memory 以後，剛才講的輸入同一個詞彙，希望它 output 不同的這個問題，就有可能被解決。

比如說，同樣是輸入 Taipei 這個詞彙，但是因為紅色 Taipei 前面接的是 leave，綠色 Taipei 前面接的是 arrive。

因為 leave 跟 arrive 它們的 vector 不一樣，所以 hidden layer 的 output 也會不同，所以存在 memory 裡面的值呢，也會不同。

即使 x2 是一模一樣的，但是因為存在 memory 裡面的值不同，所以 hidden layer 的 output 也會不一樣，所以最後的 output 也就會不一樣。


其他 Recurrent Neural Network 的架構設計

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/5d253a18-3dbf-4d35-b123-b7a7b6aa5dc1)

可以是 deep 的 Recurrent Neural Network

    比如說，我們把 x1 丟進去以後，它可以通過很多個 hidden layer，才得到最後的 output
    每一個 hidden layer 的 output 都會被存在 memory 裡面，在下一個時間點的時候呢再讀出來
    要多deep，要幾層，都可以

一些有名字的 Recurrent Neural Network
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/dff9ff8b-8e9a-4b19-8bf6-d9084eb6dccb)

Elman Network

    把 hidden layer的值存起來，在下一個時間點再讀出來
    
Jordan Network

    存的是整個 network 的 output 的值，把 output 的值存在 memory 裡面

    傳說這個可以得到比較好的 performance

        因為這邊的 hidden layer，它是沒有 target 的。
        如果有target，可以比較好控制它學到什麼樣的 hidden 的 information 放到 memory 裡面。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0e422a9b-d443-4989-b9f7-6133861127a8)

假設句子裡面的每一個詞彙我們都用 x^t 來表示它的話，它就是先讀 x^t，再讀 x^(t+1)，再讀 x^(t+2)。但是，其實它的讀取方向也可以是反過來，它可以先讀 x^(t+2)，再讀 x^(t+1)，再讀 x^t

同時 train 一個正向的 Recurrent Neural Network，又同時 train 一個逆向的 Recurrent Neural Network。把這兩個 Recurrent Neural Network 的 hidden layer 拿出來，都接給一個 output layer，得到最後的 y

舉例：在 input x^t 的時候。正向的 network 的 output 跟逆向的 network 的 output，都丟給 output layer。產生 y^t。

好處：network 看的範圍比較廣，看頭又看尾

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/8faac36b-b067-430b-8c0e-8f8b2df7b511)

LSTM 有 3 個 gate，由model自己學要打開還是關起來
  input gate

    當外界，當 neural network 的其他部分，某個 neuron 的 output 想要被寫到 memory cell 裡面的時候，必須要通過
    
    要被打開的時候，你才能夠把值寫到 memory cell 裡面去
    
  output gate

    output gate 會決定說，外界、其他的 neuron 可不可以從 memory 裡面把值讀出來
    
  forget gate

    甚麼時候 memory 要把過去記得的東西忘掉，或是它甚麼時候要把過去記得的東西 format 掉

整個 LSTM 呢，可以看成它有4個 input，1 個 output

4 個input

  想要被存到 memory cell 裡面的值

    然後它不一定存得進去，這 depend on input gate 要不要讓這個information 過去

  操控 input gate 的這個訊號

  操控 output gate 的這個訊號

  操控 forget gate 的訊號

Long Short-Term Memory (LSTM) 小小的冷知識

就是這個 dash，-，應該放在 short 跟 term 之間

只是比較長的 short-term memory

之前我們看那個 Recurrent neural network 阿，它的 memory 在每一個時間點都會被洗掉。所以這個 short-term 是非常 short

但是如果是 long short-term 的話，它可以記得比較長一點，只要 forget gate不要決定要 format 的話，它的值就會被存起來。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/bc9e5924-6207-4005-bcad-beaa871dda4e)

名詞解釋

    要被存到 cell 裡面的 input 叫做 z
    
    操控 input gate 的 signal 叫做 zi
    
    操控 forget gate 的 signal 叫做 zf
    
    操控 output gate 的 signal 叫做 zo
    
    綜合這些東西以後，最後會得到一個 output，這邊寫作 a


假設裡面一開始已經存了值 c，現在呢，假設要輸入z

  得到真正的 input

    z 通過一個 activation function得到 g(z)

    把 zi 通過 另外一個 activation function，得到 f(zi)

          假設 f(zi) = 0，input gate被關閉的時後，那 g(z) * f(zi) 就等於 0，就好像是沒有輸入一樣
          
  利用前一步的 Input，來 update memory

    把 g(z) 乘上這個 input gate 的值 f(zi) 得到 g(z)*f(zi)

    zf 這個 signal 也通過這個 activation function，得到 f(zf)

    把存在 memory 裡面的值 c 乘上 f(zf)，得到 c*f(zf)

    把c*f(zf) 加上 g(z)*f(zi) 得到 c'，存回去memory

          假設 f(zf) 是 0， forget gate 被關閉的時候， 0 乘上 c 還是 0，也就是過去存在 memory 裡面的值呢，就會被遺忘
  
  得到真正的output a

    把這個 c' 通過 h，得到 h(c')
  
    zo 通過 f 得到 f(zo) 跟這個 h(c') 乘起來，得到 h(c') * f(zo)，也就是最後的 a

  這3個 zi, zf, zo 他們通過的這3個 activation function f ，通常我們會選擇 sigmoid function

    sigmoid 的值是介在 0~1 之間的，而這個 0~1 之間的，代表了這個 gate 被打開的程度
    如果這個 f 的 output 這個 activation function 的 output 是 1，代表這個 gate 是處於被打開的狀態，反之呢，代表這個 gate 是被關起來的

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0fa23d87-8589-47f5-ae06-3f16f8e78235)

介紹

    在 network 裡面，只有一個 LSTM 的 cell

    那我們的 input 都是三維的 vector

    output 都是一維的 vector

Memory Gate模擬

    第二個 dimension x2 的值是 1 的時候

        x1 的值就會被寫到 memory 裡面去

    假設 x2 的值是 -1 的時候

        memory 就會被 reset

Output Gate模擬
    假設 x3 等於 1 的時候

        你才會把 output gate 打開，才能夠看到輸出

手爆的完整過程查看影片比較清楚

跟原本RNN的關係

  原來的 neural network 裡面會有很多的 neuron

    我們會把 input 乘上很多不同的 weight，然後當作是不同 neuron 的輸入

    然後每一個 neuron 它都是一個 function。它輸入一個 scalar，output 另外一個 scalar

LSTM

    把那個 LSTM 的那個 memory cell 想成是一個 neuron 就好

    做的事情只是把原來一個簡單的 neuron，換成一個 LSTM 的 cell

    而現在的 input 它會乘上不同的 weight，當作 LSTM 的不同的輸入

怎麼得到LSTM的四個input？現在假設只有兩個neuron：

    那 x1, x2乘上某一組 weight，會去操控第一個 LSTM 的 output gate

    乘上另外一組 weight，操控第一個 LSTM 的input gate

    乘上一組 weight，當作第一個 LSTM的input

    乘上另外一組 weight，當作另外一個 LSTM的forget gate的 input
 
在原來的 neural network 裡面，一個 neuron 就是一個 input，一個 output，

在 LSTM 裡面它需要 4 個 input，它才能夠產生一個 output


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3fd7e466-ccf3-43c6-a5fd-5061a72a2d0b)

一般的 neural network 只需要部分的參數

LSTM 還要操控另外三個 gate，所以他需要 4 倍的參數
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b983ba35-f464-4684-b34a-d5d3a50daf07)

假設我們現在有一整排的 neuron，假設有一整排的 LSTM

那這一整排的 LSTM 裡面，每一個 LSTM 的 cell，它裡面都存了一個 scalar。把所有的 scalar 接起來，就變成一個 vector，這邊寫成 c^(t-1) （橘色）

這邊每一個 memory 它裡面存的 scalar，就是代表這個 vector 裡面的一個 dimension

先得到一整排 input z

    在時間點 t，input 一個 vector, x^t，這個 vector，它會先乘上一個 linear 的 transform，變成另外一個 vector z

    z 這個 vector 的每一個 dimension 呢，就代表了操控每一個 LSTM 的 input。所以 z 它的 dimension 就正好是 LSTM 的 memory cell 的數目。
    
再得到一整排 input gate z^i

    這個 x^t 會再乘上另外一個 transform，得到 z^i

    這個 z^i 呢，它的 dimension 也跟 cell 的數目一樣，每一個 dimension都會去操控一個 memory

再得到一整排 forget gate z^f，一整排 output gate z^o

這 4 個 vector 的 dimension 都跟 cell 的數目是一樣的，那這 4 個 vector 合起來就會去操控這些 memory cell 的運作

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/94f917e6-664a-4235-8b88-b1a1d981f4ff)

現在 input 分別是 z, z^i, z^f, z^o，注意一下就是這 4 個 z 其實都是 vector

所有的 cell 是可以共同一起被運算的，怎麼一起共同被運算呢

    把 z^i 先通過 activation function，然後把它跟 z 相乘

        這個乘呢，是這個 element-wise 的 product 的意思

    這個 z^f 也要通過 forget gate 的 activation function的值，跟之前已經存在 cell 裡面的值，兩者相乘，存回memory

    把上述這兩個值加起來（ z^i 跟 z 相乘的值加上 z^f 跟 c^(t-1) 相乘的值）

    把 z^o 通過 activation function的值， 跟相加以後的結果再相乘，就得到最後的 output 的 y

（上述的過程跟單一cell都差不多，只是這次是一整排一起做，直接矩陣相乘）

LSTM 的最終型態

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d7d623d1-9feb-430c-ac58-4c1467332496)

以上只是一個 simplified 的 version

真正的 LSTM 會怎麼做呢：

    把前一個 hidden layer 的輸出接進來當作下一個時間點的 input（如上圖紅色線，指到的c）

    還會加一個東西呢，叫做 "peephole"

        把存在 memory cell 裡面的值也拉過來 （上圖藍色h）

讓人傻眼的複雜模型：

    同時考慮了 x, 同時考慮了 h, 同時考慮了 c

    把這 3 個 vector 並在一起，乘上4個不同的 transform，得到這4個不同的 vector，再去操控 LSTM
    
    通常不會只有一層，都胡亂疊個五、六層

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/53dd1f2a-e565-4d1e-852e-1818600792f0)

Keras 支援三種 Recurrent neural networks

    一個是 LSTM

    一個是 GRU

        LSTM 的一個稍微簡化的版本，它只有兩個 gate
        
        performance 跟 LSTM 差不多，而且少了 1/3 的參數，所以比較不容易 over-fitting

    一個是 SimpleRNN


如果要使用最前面讲的最简单的RNN，请参考SimpleRNN.py，代码大部分解释可以看LSTM，示例中，我们将使用一个合成的时间序列数据，该数据遵循一个已知的数学模式，并添加一些随机噪声来模拟真实世界数据的不确定性。展示了如何使用 SimpleRNN 处理时间序列数据。预测点（红色点）展示了模型基于过去观测到的数据模式对未来值的预测。在实际应用中，可以将此方法应用于更复杂的时间序列数据，如金融市场数据、气象记录等。此外，模型的性能可通过调整 RNN 单元的数量、训练轮数或学习率等参数进一步优化。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/8de68970-eed3-4b3f-bb62-c33385523b9b)


GRU就是简化版的LSTM，在这个示例中，我们将使用门控循环单元（GRU）来处理一个天气温度预测的问题，这是时间序列预测中的一个常见应用。GRU 是 RNN 的一种变体，它通过引入更新门和重置门来解决传统 RNN 在长序列上训练时的梯度消失问题。用过去 24 小时的温度数据来预测下一小时的温度。您可以看到模型预测的温度与实际温度之间的关系。理想情况下，预测曲线应紧跟实际温度曲线，这表明模型能够准确捕捉温度变化的模式。如果预测曲线与实际数据有较大偏差，可能需要调整模型参数、增加训练轮数或重新考虑输入特征。每个时间点都有大量的数据点（例如，频繁的预测值与实际值），这些点在可视化时可能会相互重叠，形成看似连续的区间而不是离散的点。在图中，红色线相对平滑，并且几乎完全覆盖了蓝色线，显示了模型预测与实际温度之间高度的一致性。请参考GRU.py
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/92463479-fa37-4bdb-9ccd-5642b40a9c71)


为了保持例子的简便性，我们将使用模拟的天气数据，但这个方法可以直接应用于真实的气象数据。

最后一个就是LSTM，在本示例中，我们将使用 LSTM 来预测股票价格，这是一个典型的时间序列预测问题。我们将使用 TensorFlow 和 Keras 构建模型，并使用随机数据来模拟股票价格数据。我们可以看到下图，一个是随机数据来模拟股票价格数据进行训练和测试，最后可以看到模型是如何预测接下来的股票价格走势。请参考LSTM.py
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0d9241d5-0abc-4ce8-9bfa-5ddc55fc910b)
