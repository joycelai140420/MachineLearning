半监督学习（Semi-supervised Learning）简介
半监督学习是介于监督学习和无监督学习之间的一种机器学习方法。与传统的监督学习仅使用标记的数据不同，半监督学习同时利用大量未标记的数据和少量标记的数据进行模型训练。它基于这样一个假设：即使没有标签，数据的分布本身也可以提供学习过程中的有价值信息。

Transductive learning: unlabeled data 來自於 testing data
Inductive learning: unlabeled data 並非來自於 testing data

应用

图像和语音识别：在图像和语音数据中，获取未标记数据通常比较容易，而标记数据则需要耗费更多资源。不过图片有些可以透过翻转等方式复制图片增加数据，语音可以透过变声器来增加数据等等。

文本分类：对文档进行分类时，未标记文本通常丰富且易于获取，而对文本的手动标记则代价较高。

生物信息学：在生物信息学中，通常可以得到大量的生物数据，但相关的标注（如功能标注）可能不足。

优点

有效利用未标记数据：可以显著降低获取标记数据的成本，特别是在数据标注成本高昂的领域。

提高模型性能：在标记数据稀缺的情况下，半监督学习可以通过未标记数据学习数据结构，提高模型的泛化能力。

灵活性：可以应用于小样本学习场景，减轻对大量标记数据的依赖。

缺点

假设依赖性：大多数半监督学习算法基于一定的假设（如业务的假设设定算法等），如果这些假设与实际情况不匹配，可能导致性能下降。

算法复杂性：相较于传统监督学习，半监督学习算法通常更加复杂，需要更细致的调参和算法选择。

标记数据不平衡问题：如果少量的标记数据不足以代表整个数据集的分布，可能会导致模型的偏差。

接下来就是截取台大Hung-yi Lee老师课程内容

例子
假設現在要做一個辨別貓狗的圖片分類器，同時有一大堆有關貓跟狗的圖片，而這些圖片大部分是沒有 label 的，如下圖：

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b869a91a-b5a2-4d61-9449-f58febce3f75)

假設我們只考慮有 label 的 data (藍點和橘點) 的話，那分界可能會很直覺地畫出圖上鉛直的紅線，但是假如納入 unlabeled data (灰點) 的分布，可能直覺就會讓人改成畫出圖上的斜線作為分界。

使用 unlabeled 的方式往往伴隨著一些假設，所以 semi-supervised learning 有沒有用就取決於假設符不符合實際，有可能假設不夠精確，左下角的灰點實際上是一隻狗，只是因為背景同樣是綠色而靠近貓 labeled 的 data，因此使用 semi-supervised learning 需要在正確的假設下。

Generative Model Supervised learning

在 Supervised learning 中，有一堆訓練資料，它們的 label 都是已知的。

以下圖為例，已知分別屬於 class 1 或 class 2，在這個情況下去估測 class 1 和 class 2 的 prior probability，如果我們用的是 Gaussian distribution 的假設，那估測的就是，這個 class 1 是從 mean = μ^1, covariance = Σ 的 Gaussian distribution，及 class 2 從 mean = μ^2, covariance = Σ 的 Gaussian distribution。於是有一個新的 data 時，便可以計算它的 posterior probability 來做分類。

共用 Gaussian 的 covariance 可能會有較好的表現在之前已經提過了
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/33f47ae8-415a-40c9-aacc-bce23d6c5d99)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/76bd8795-e8ed-4ef2-a210-5a36ebba2cc9)

如果資料的分布情形如上圖所示，則之前 supervised 所得到的結果並不合理，實際的 covariance 應該要更接近圓圈的形狀，實際的 μ^2 應該在下方一點的位置，而非 labeled data ，那是因為 sample 偏差的結果，此外根據 unlabeled data，class 2 的 prior probability 應該是較大的。

请参考其Generative Model Supervised learning.py范例
