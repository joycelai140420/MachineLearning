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

Entropy-based Regularization

另外一個方法考慮的是輸出結果的 distribution，期望在 unlabeled data 上能得到集中而非發散的分布，因為現在有著 low-density seperation 的假設。（非黑即白意味分类output要很集中，分布如下图的Good图示）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/81b979d5-7cea-4803-af53-598bed762f7a)
，
根据bad的图，分布散得比较大，entropy比较大，我们希望在labeled data 要分类比较正确，希望在unlabeled data上output 、entropy要越小越好，所以根据这个假设，我希望labeled data的model的output跟正确的output，两个的距离要越近越好。用cross entropy来evaluate他们之间的距离。然后再unlabeled data的部分会加上他的output distribution的entropy且希望值越小越好。可以*上lambda来控制权重要偏向unlabeled da多一点还是labeled datalabeled data。

请参考其Entropy-based Regularization.py范例

Semi-supervised SVM
SVM 的運作方式就是在兩個 class 的 data 中，找一個邊界 (margin) 最大的分隔線 (boundry)，讓這兩個 class 的資料分的越開越好，與此同時，它也要有最小的分類的錯誤。

在有一些 unlabeled data 的清況下，Semi-supervised SVM 的做法是窮舉所有可能的 label，接著對每一個可能的結果，都用 SVM 來訓練一個模型，最後，再去檢查哪一個 unlabeled data 的可能性在窮舉所有的可能的 label 裡面，可以讓邊界最大、同時又最小化 error。（如下圖）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/886d648a-c5cd-4b45-96e7-fbc4f16c5d0f)

窮舉所有 unlabeled data 的 label 直覺上會讓計算量爆炸到不可行的程度，例如有一萬筆 unlabeled data ，就需要窮舉 2 的一萬次方，根本毫無可能，於是上圖 reference 的 paper 就提出了一個很近似的方法。Thorsten Joachims 提出了一种称为 TSVM（Transductive Support Vector Machines）的半监督学习方法，也被称为 TSVM 或 S3VM，用于处理同时有标记和未标记数据的情况。基本精神是是你一開始先得到一些 label，然後每次改一筆 unlabeled data 的 label，看看能否讓 objective function 變大，如果變大就繼續用這個 label。

Smoothness Assumption（近朱则赤，近墨则黑）

在半监督学习中，平滑性假设（Smoothness Assumption）是指如果两个样本在特征空间中是接近的，那么它们对应的输出标签也应该是相似的。在这个假设下，我们可以利用未标记数据的分布来预测它们的标签，进而增强学习算法的性能。

這個假設簡單來說就是：如果 x 是像的，那它們的 label y 也就像，這個假設聽起來沒有甚麼，而且這麼講這個假設其實是不精確的，因為一個正常的 model 如果它不是很深的話，一個接近的 input，自然會有接近的 output。

這麼說仍有些難懂，以下圖為例：雖然 x^2, x^3 在距離上是接近的，但由於 x^1, x^2 在同一個密度高的區域裡，因此在 Smoothness assumption 下，x^1, x^2 有更高的機率是同一個 label。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f4c25100-6ddf-41de-b53b-6d8787ebce97)

這個方法在文件分類上可能非常有用。假設你現在要分天文學跟旅遊的文章，那天文學的文章有它固定的 word distribution，像是會出現 asteroid, bright，而旅遊的文章會出現 yellow stone 等等。如果今天你的 unlabeled data 跟你的 labeled data 是有重疊的，那很容易處理這個問題，但是在真實的情況下， unlabeled data 跟 labeled data 中間可能沒有任何重疊的 word，因為一篇文章的篇幅往往有限。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/54f46606-4023-4cd7-b359-d9a9f9d5001e)

這種情形下，如果收集到夠多的 unlabeled data，就可以說 d1 跟 d5 相近，d5 又跟 d6 像，就可以一路 propagate 過去。

Cluster and then Label
實踐 Smoothness assumption 最簡單的做法是 Cluster and then label，這個方法太簡單以至於沒什麼可以談的。做法是先將資料做分群 (clustering)，接著看各個 cluster 中哪個 label 的數量最多，便將該 cluster 的 unlabeled data 視作該個 label，最後拿這樣的資料去做訓練。

這個方法會發揮作用的假設就是你可以把同一個 class 的東西 cluster 在一起，可是在圖片裡面要把同一個 class 的東西 cluster 在一起其實是沒那麼容易的，像是以前在講為甚麼要用 deep learning 的時候，提過不同 class 可能會長得很像、同一個 class 可能會長得很不像。

教授在此提及了用 Deep autoencoder 再做 Cluster and the label，但尚未教到 Deep autoencoder，因此先略過不提。

在半监督学习中，平滑性假设（Smoothness Assumption）是一个普遍的假设，它认为在高密度区域的决策边界应该是平滑的，而不是在低密度区域。换句话说，如果两个样本在特征空间中彼此非常接近，它们很可能属于同一个类别。

Graph-based Approach

用 Graph 來表達也就是說，要想辦法資料點之間的 edge 建出來，有了這個 graph 以後，所謂的 high density path 的意思就是兩個點在這個 graph 上面是相連的，走得到的就是同一個 class。

怎麼建一個 graph 呢？

有些時候是很自然就可以得到，假設現在要做的是網頁的分類，而你有記錄網頁有網頁之間的 hyperlink，這就很自然地表示了網頁間是如何連結的；或者現在要做的是論文的分類，而論文和論文之間有引用的關係，這個關係式也是另外一種 graph 的 edge。

當然有時候，你需要自己想辦法建 graph，而 graph 的好壞對結果影響是非常大。不過這就非常的 heuristic （憑經驗），藉由經驗和直覺去選擇比較好的做法。

通常的做法是如此的：先定義兩個 object 之間怎麼算它們的相似度（例如影像就適合用 autoencoder 抽出來的 feature 算相似度），算完相似度以後就可以建 graph 了。

建 Graph 的方法有很多種：

K nearest Neighbor 假設令 k = 3，每一個 point 都跟它最近的、相似度最像的 3 個點做相連。
e-Neighborhood 每一個點只有跟它相似度超過某一個 threshold e 的那些點做相連

所謂的 edge 也不是只有相連和不相連這樣二元的選擇而已，edge 可以擁有 weight 來表示被連接起來的兩個點之間的相似度。建議比較好的選擇是用RBM Function 來計算相似度
![1714397527416](https://github.com/joycelai140420/MachineLearning/assets/167413809/58ecb8a5-afc2-48c3-b72e-f01b066e121c)

所以 graph-based 方法的精神是，假設有兩筆資料點是屬於 class 1，和他相鄰的點是 class 1 的機率也會上升，而這個機率會在圖上連通的區域傳遞下去。因此使用 graph-based 的方法需要有足夠的資料量，要是資料量不夠大，便有可能發生 label 無法 propagate 下去的可能。

Definition of Smoothness of the Labels on the Graph
![1714397527416](https://github.com/joycelai140420/MachineLearning/assets/167413809/10e58051-9b97-4e3c-b1ff-6f27524ba7cf)

接下來要定義在 Graph-based 的方法下，label 有多符合 Smoothness assumption 這個假設，是加總所有的 data pair 的相似度乘上他們 label 的差異，數值越小代表越 smooth（舉例如下圖）。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/912620e6-7fee-4061-82b2-e5b2111feb49)


