Auto-encoder的基本结构
Auto-encoder 由两部分组成：编码器（Encoder）和解码器（Decoder）。

编码器：

编码器的作用是将输入数据压缩成一个更小的编码。这个过程中数据的维度被减少，因此编码器需要学习如何最有效地捕捉输入数据的关键特征。
编码器通常由一系列降维步骤组成，这些步骤可以是简单的线性层（如全连接层），或者包括更复杂的结构，如卷积层（在处理图像数据时）。

解码器：

解码器的目标是从编码中重构输入数据，尽可能地近似原始输入。
解码器结构通常是编码器的镜像，包含逐步增加数据维度的层，直到达到原始输入的大小。

工作原理

Auto-encoder 通过最小化重构误差（即输入和输出之间的差异）来训练。
常用的误差函数包括均方误差（MSE）或交叉熵损失，这取决于数据的特性和分布。
训练过程中，网络学习到的编码代表了数据的一种压缩表示，该表示尝试捕获数据的最重要特征。

变体和应用
Auto-encoder 有多种变体，包括：
稀疏自动编码器（Sparse Autoencoder）：在编码器的激活中引入稀疏性，迫使模型学习更有信息量的特征。
去噪自动编码器（Denoising Autoencoder）：通过训练网络从带噪声的输入中重构原始无噪声输入，增强模型的鲁棒性和特征学习能力。
变分自动编码器（Variational Autoencoder, VAE）：这是一种生成模型，它不仅学习编码和解码，还学习输入数据的潜在分布。

Auto-encoder 的关键在于它的编码能力，可以从数据中学习到有用的特征，这使得它在无监督学习领域中非常有价值。通过自动编码器学到的特征通常比原始数据更容易处理，可以用于各种数据分析和机器学习应用。

以下是参考台大老师Hung-yi Lee的授课内容

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a57bbacf-1868-4709-beeb-70717653d84b)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a5e52ac1-953c-4305-bf45-826e34b5c91b)

PCA 只有一個 hidden layer，而我們可以有更多 hidden layer
output 是 x^，希望這個 x 跟 x^ 越接近越好
training: back propagation
從 input 到 bottleneck layer 的部分就是 encoder
從 bottleneck layer 的 output 到最後的x^， 到最後整個 network 的 output 就是 decoder

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7663c314-412d-4ee8-a235-ab6ce7d9347e)

PCA：
从784 維降到 30 維，再從 30 維 reconstruct 回 784 維，比較模糊的

deep auto-encoder：
从784 維，先擴展成 1000 維，再把 1000 維降到 500 維再降到 250 維再降到 30 維，再把 30 維變成 250 維再變成 500 維 1000 維，再解回來 784 維，比較清楚

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/cc625271-5f45-451a-a231-bfee1d4c94b0)

PCA：混在一起

deep auto-encoder：是分開的，不同的數字會變一群一群
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ef2801ac-806f-40d6-a6ff-d2de459ceaba)

Use on Text Preprocessing
文字（文章）搜尋

1.vector space model：
文章都表示成空間中的一個 vector

2.計算輸入的查詢詞彙跟每一篇 document 之間的 inner product 或是 cosine similarity 等等

3.根據值的大小決定是否 retrieve

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/8c29b60f-e2a6-45ff-a380-8f5e4fb50497)

把一個 document 變成一個 vector
bag-of-word

1.最 trivial
2.Term Frequency
3.Inverse Document Frequency
4.TFIDF
5.沒辦法考慮任何語意相關的東西，每一個詞彙都是 independent

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1b6cc01d-023b-4492-b710-516e95727b56)

auto-encoder

input: document
query: 一段文字

用比較小的 lexicon size，把一個 document 就把它變成一個 vector，再把這個 vector 通過一個 encoder，把他壓成二維，然後 Visualize

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/039766d2-cf9e-4412-9a80-c88cc67cc8ff)

發現同一類的 document，就都集中在一起，散佈像一朵花一樣

要做搜尋的時候

1.輸入一個詞彙、查詢詞
2.把 query 也通過這個 encoder，把他變成一個二維的 vector
3.看query 落在哪邊，就可以知道說這個 query 是哪個 topic

我们也可以用来找图
Use on Image Searching
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/09ba2280-200e-4450-9295-0df5959ac125)

以圖找圖。
最簡單的方法，在 pixel wise 上做比較，但是找不到好的結果的。哈哈

Auto-encoder for CNN
應該要用 deep auto-encoder 把每一張 image 變成一個 code，然後在 code 上面再去做搜尋
因為這是 unsupervised，train 這種 auto-encoder 的 data 是永遠不缺的

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/76a995ce-0132-47de-9253-bf294d1e0d6d)

在這種 code 上面算相似度的話，就會得到比較好的結果

如果 encoder 的部分是做 convolution 再做pooling，convolution 再做pooling 理論上 decoder 應該就是做跟 encode 相反的事情 本來有 pooling 就做 unpooling，本來有 convolution 就做 deconvolution

Unpooling

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e63825bc-2125-42a5-905b-422fe4328325)

要把原來比較小的 matrix 擴大

方法一：
記住pooing是從哪裡取值，照樣還原回去，沒有取值的部分捕0

方法二：
直接把那個值複製四份，不用去記從哪裡取 (Keras 是用這種方式)


Using with Pre-training
有時候在煩惱怎麼做參數的 initialization，這種找比較好的 initialization 方法，就叫做 pre-training。那可以用 auto-encoder 來做 pre-training

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/6e73cfea-59da-496c-9c75-7d49b61b7bb5)

先 train 784-1000-784，把 weight(W_1) 記下來

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a5ef9649-f370-48b6-8a54-b2e2512a9ac7)

再 train 784-1000-1000-1000 784-1000 部分的 weight(W_1) 用前面的，並 fix 住 一樣把 weight(W_2) 記下來

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e24ff365-e463-42b4-8c07-311946690fd0)

再 train 784-1000-1000-500-1000 784-1000 部分的 weight(W_1) 以及1000-1000 部分的 weight(W_2) 用前面的，並 fix 住 一樣把 weight(W_3) 記下來

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/8497d6c8-2ff6-429a-953e-ad79c07948d8)

就可以整個串起來，並把剛剛那些 weight (W_1, W_2, W_3) 作為 model 的 initialization，再 random initialize 最後 500 到 100 的 weight，再用 back propagation 去調一遍，我們稱之為 fine tune。

你可以参考Auto-encoder for CNN.py,上面是原始结果下面是过编码器又过解码器部分结果。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/266c46e1-0661-4c7d-93bf-8fc8132ad9ce)


其实也有其他的变形也有De-noising auto-encoder 或Contractive auto-encoder，也就是在input 加上noise 后也要学得很好，过滤出杂讯。

Generating Outputs

1.那個 decoder 其實是有妙用的，可以拿 decoder 來產生新的 image
2.也就是說我們把 learn 好的 decoder 拿出來，然後給他一個 random 的 input number，output 希望就是一張圖


這件事可以做到嗎，其實這件事做起來相當容易

1.在MNIST dataset上，把每一張圖，784 維的 image 通過一個 hidden layer 然後 project 到二維
2.再把二維通過一個 hidden layer 解回原來的 image
3.那在 encoder 的部分，那個二維的 vector 畫出來長這樣

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1ca65306-622d-419c-8d14-6713f22c0133)

1.在紅色這個框框裡面等間隔的去 sample 一個二維的 vector 出來

2.然後把那個二維的 vector 丟到 NN decoder 裡面， output 一個 image 出來

3.可以發現很多有趣的現象
  從下到上，感覺是圓圈然後慢慢的就垮了
  右下這邊本來是不知道是四還是九，然後變八
  再往上然後越來越細，變成 1
  最後不知道為什麼變成 2，還蠻有趣的

然后會發現在這邊感覺比較差，是因為在這邊其實是沒有 image，所以你在 input image 的時候其實不會對到這邊。 這個區域的 vector sample 出來，通過 decoder 他解回來不是 image。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/681c273c-c347-458b-8188-1ba259a5b45b)

所以要 Sample 到一個好的地方 因為我必須要先觀察一下二維 vector 的分佈，才能知道哪邊是有值的，才知道從那個地方 sample 出來比較有可能是一個 image。

可是這樣你要先分析二維的 code 感覺有點麻煩，有個很簡單的做法就是在你的 code 上面加 regularization。在你的 code 直接加上 L2 的 regularization，讓所有的 code 都比較接近零。接下來就在零附近 sample就好了。

接下來我就以零為中心，然後等距的在這個紅框內 sample image，sample 出來就這個樣子。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3101a8f0-e879-49bd-86f0-c027626e1ff8)

從這邊你就可以觀察到很多有趣的現象 會發現說：

這個 dimension 是有意義的：
    從左到右橫軸代表的是有沒有圈圈
    縱的呢，本來是正的，然後慢慢就倒過來
    
所以你可以不只是做 encode，還可以用 code 來畫。這個 image 並不是從原來 image database sample 出來的，他是 machine 自己畫出來的。

