以下文字内容与投影片是来自于台大老师Hung-yi Lee授课内容

Unsupervised Learning - Deep Generative Model (Part II)

Why VAE
Intuitive Reason
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2c952460-c1a2-4a7f-b16b-e6e4e2e575bf)

在 VAE 裡，sigma 的作用是作為 noise（取指數確保為正），讓 code 「c」 在即便有 noise 的情況下仍能生成如我們所想的結果（「e」 是從 normal distribution 0 到 1 中取出的向量），但如果直接如此對模型做訓練，機器自然會產出 variance 為零，使用沒有雜訊的 code 給 decoder，因此要再加上上圖左下角的 loss function（包含一個 L2 regularization）。下圖的紅線和藍線代表的即是 exp(sigma_i) 以及 (1+sigma_i)，而綠線則是他們相減的值，如此就能確保電腦不會一昧將 variance 壓低至零。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a64cd4af-d184-453b-80df-9730555bec17)

Formal Explanation
假設一張 20*20 的圖片，它在高維的空間中也就是一個 400 維的點，那我們現在要做的其實就是估計高維空間上的機率分佈 - P(x)。

現在有一個複雜的 distribution，如下圖的黑色線，今天只要 Gaussian 的數目夠多就可以產生很複雜的機率分佈。雖然黑色很複雜，但它背後其實是有很多 Gaussian 疊合起來的結果。

那這個式子怎麼寫它呢？

首先，如果你要從 P(x) sample 一個東西的時候，你要先決定你要從哪一個 Gaussian sample，也就是根據每一個 Gaussian 的 weight P(m) 去決定你要從哪一個 Gaussian 做 sample，然後再從你選擇的那個 Gaussian 裡面 sample data P(x|m)。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2b93c999-0aab-4eca-a701-b628ad7b2f60)

這件事情很像是做 classification 時，做 clustering 其實是不夠的，更好的表示方式是用 distributed 的 representation，也就是說每一個 x 有一個 vector 來描述它的各個不同面相的 attribute 描述它各個不同面向的特質。所以 VAE 其實就是 Gaussian mixture model 的 distributive representation 的版本，怎麼說？

首先從一個 normal distribution sample 一個 z 出來，根據 z 可以決定 μ 跟 σ，我們知道說 neural network 就是一個 function，所以你可以說我就是 traing 一個 neural network 輸入 z 接著輸出兩個 vector μ 跟 σ，所以 x 的機率就成了 P(x) = int limits_z P(z)P(x|z) dz。而 P(z) 也不一定要是 normal distribution，可以自行決定成任合一種分佈，不過 normal distribution 也的確是合理的假設，並且因為神經網路是極為強大的 function，所以只要 nueron 夠多就能表示任何 P(x)。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/5cacfe7e-84b5-423d-83a7-aba66544f3c8)

尋找從 z 得出 mean 和 variance 的神經網路要怎麼訓練呢？它的目標就是要最大化 likelihood，也就是圖中的 L。另外有一個分佈 q(z|x)，它是由 x 決定在 z 這個空間上面的 mean 跟 variance，同樣由神經網路構成。這兩個 function 就是 VAE 裡的 decoder 和 encoder。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/85da4e5d-4d2c-407c-968f-4c79942a5b52)

Likelihood 經過推導之後便會得到 KL divergence 和左邊的積分式，由於 KL divergence 是兩個分部間的距離一定會大於零，於是我們改稱左邊的積分式為 lowerbound L_b，在改寫後我們要做的事便是找 P(x|z) 和 q(z|x)。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/445c9f2f-e6d7-4949-955a-059f349ee3ef)

因此，NN 做的事情就是在最小化 P(z) 跟 q(z|x) 的 KL divergence，以及最大化一項積分式，這兩件事合起來就是 auto-encoder。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/392aa9d5-6d3d-4870-a534-6cfc2b42d21b)


Conditional VAE
假設在做數字辨識，我們在 train VAE 的時候除了給它這是什麼數字，也給他要的 style，我們就能利用一個數字來生成跟它風格相近的其他數字。

Problems of VAE
VAE 其實它從來沒有去真的學怎麼產生一張看起來像真的圖片，因為它所學到的事情是它想要產生一張圖片跟在 database 裡面某張圖片越接近越好。

比如說我們用 mean square error 來計算兩張圖片間的相似度，此時一個 pixel 的差距落在不一樣的位置，其實是會得到非常不一樣的結果，但是對 VAE 來說都是一個 pixel 的差異，這兩張圖片是一樣的好或一樣的不好，所以 VAE 所產生出來的圖片往往都是 database 裡面的圖片的 linear combination 而已。

Generative Adversarial Network
Generative adverserial network (GAN) 基本的概念就是有兩個神經網路，一個負責從零開始生成擬真的結果要騙過另外一個神經網路，另外一個則看過真實的資料並想辦法變聰明不被騙過，兩個就迭代的不停進步。前者就是 GAN 之中的 generator，後者則是 discriminator。

以圖片生成為例，GAN 的訓練方法就是：

1.首先，Generator 會根據從某個分佈中取樣出的一個 vector 來生成一些假的圖片。
2.接著 discriminator 生成的圖片標為 0 (fake) 以及原有的 labeled data 去調整他的參數。
3.根據第一代的 discriminator，我們把 generator 加 discriminator 視為一個很大的神經網路，並利用梯度下降 (gradient descent) 來做 back-propagation 以最大化輸出，這個動作的意義即是盡可能讓 discriminator 認為生成的圖片是真實的，值得注意的是此處只能調整前半段的參數 (generator)。
4.再回到第一步回去利用進步的 generator 來訓練 discriminator。
