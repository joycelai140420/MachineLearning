
无监督学习中的线性方法简介

无监督学习的线性方法涉及使用线性模型来发现数据的结构和分布，而无需预先标记的数据。这类方法通常用于数据降维、聚类、异常检测等。

常见线性方法

主成分分析 (PCA)：通过正交变换将数据投影到低维空间，保留最大方差的方向，用于降维。

因子分析：假设观测变量由潜在变量和特定噪声生成，用于发现潜在变量。

独立成分分析 (ICA)：将多变量信号分解成相互独立的非高斯信号源，用于信号分离。

线性判别分析 (LDA)：虽然 LDA 通常用于监督学习，但也可以用于数据降维。

优点

可解释性：线性方法由于其简单性，通常更容易解释。

计算效率：相比于非线性方法，线性方法计算上通常更快。

理论基础：有着坚实的理论基础，且实现简单。

缺点

线性假设：假设数据具有线性结构，这可能不适用于复杂的、非线性分布的数据。

有限的复杂性：可能无法捕捉更复杂的数据模式。

对异常值敏感：特别是 PCA，异常值会影响主成分的计算。

应用场景

降维：在高维数据上进行可视化或预处理。

数据预处理：去除噪声，发现有用的信号。

特征提取：在聚类或分类之前找到表示数据的新特征。

探索性数据分析：寻找数据中的模式和关系。

在训练时提高准确率的方法

数据预处理：标准化或正则化数据，确保所有特征在相同的尺度上。

异常值处理：识别和处理异常值，因为它们可能对模型产生不利影响。

超参数优化：调整模型的超参数以获得最佳性能。

在测试集上提高准确率的方法

在无监督学习中，通常没有“测试集”这个概念，因为我们不会进行预测任务。不过，我们可以使用以下策略来评估模型的性能：
      轮廓分数 (Silhouette Score)：评估聚类的质量。
      重构误差：评估降维后数据与原始数据的一致性。
      
提高模型准确率的其他动作

      增量学习：如果数据太大无法一次性加载，可逐步加载数据并更新模型。
      多模型融合：结合多种无监督学习方法的结果。
      模型校准：在特定应用中，可以使用一些有标记的数据对无监督模型进行校准或验证。  

在训练无监督学习模型时，关键是要理解和可视化数据，以及识别数据中可能存在的结构和模式。有效地使用数据的自然结构可以帮助设计更好的特征和模型，进而提高数据分析的准确性。

一下就是来自于台大老师Hung-yi Lee 授课部分内容：

Linear Method Dimension Reduction

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7cc7fe3e-5293-4480-8769-095e5df1b71f)

Dimension Reduction 分成兩種：

一種做的事情，叫做化繁為簡。
那另外一個是 Generation，也就是無中生有。

化繁為簡
又可以分成兩大類，

一種是做 Clustering
一種是做 Dimension Reduction
所謂的化繁為簡的意思是說現在有很多種不同的 input，找一個 function它可以 input 看起來像樹的東西，output 則是抽象的樹。

也就是把本來比較複雜的 input，變成比較簡單的 output。

畢竟在做 Unsupervised Learning 的時候通常只會有 function 的其中一邊，所能擁有的 training data，就只有一大堆的 image，不知道說它的 output，應該要是長什麼樣子。

無中生有
那另外一個 Unsupervised Learning 會做的事情呢。

要找一個 function，隨機給它一個 input。比如說輸入一個數字 1，然後，它就會 output 這棵樹; 輸入數字 2，就 output 另外一棵樹; 輸入數字 3，就 output 另外一棵。

在這個 task 裡面，要找的是這個可以畫圖的 function，只有這一個 function 的 output，沒有這一個 function 的 input。

只有一大堆的 image，但是不知道輸入怎麼樣的 code，才可以得到這些 image。

這一份講義主要著重於在 linear 的 Dimension Reduction上。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/92800ba6-d3b7-4226-9a90-16363fcabac2)

什麼是 clustering 呢?
clustering 就是有一大堆的 image，假設現在要做clustering，就是把它們分成一類、一類、一類的。

把本來有些不同的 image，用同一個 class 來表示就可以做到化繁為簡這一件事情。

到底應該要有幾個 cluster，就跟 neural network 要幾層一樣，是 empirical 地去決定。

在无监督学习中，聚类是一种常用的技术，用于发现数据中的自然分组。K-means 是最流行的聚类算法之一，它的目标是将数据点划分为 K 个群组，使得每个数据点与其所属群组的中心（即质心）的距离之和最小。

请参考KMeans.py 其中一个是使用sklearn toolkit范例，一个是用numpy 来深入了解其算法运作原理。

Hierarchical Agglomerative Clustering (HAC)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b69b7f45-0d3b-49f2-ac82-30aca5e97271)

clustering 有另外一個方法叫做，Hierarchical Agglomerative Clustering (HAC)。

這個方法，是先建一個 tree。

假設你現在有 5 個 example：

把這 5 個 example兩兩、兩兩去算他的相似度，然後挑最相似的那一個 pair 出來。（假設現在最相似的 pair，是第一個和第二個 example，那就把第一個 example 和第二個 example merge 起來。比如說，把它們平均起來，得到一個新的 vector。）
接下來，只剩下 4 筆 data，把這 4 筆 data，再兩兩去算它的相似度，再把它們 merge 起來，把它們平均起來得到另外一筆 data。
現在只剩三筆 data，再去兩兩算它的 singularity。（然後，黃色這一個和中間這一個最像，就再把它們平均起來）
最後只剩紅色跟綠色，只好把它平均起來。
就得到這個 tree 的 root，就根據這 5 筆 data，它們之間的相似度，建立出一個 tree structure。

這個 tree structure 可以告訴我們哪些 example 是比較像的，比較早分支代表比較不像，因此可以根據想要分的cluster數目，對這個tree做拆解。

K-means VS HAC
HAC 跟 K-means最大的差別是如何決定cluster 的數目。

在 K-means 裡面，要事先決定你 K 的 value 是多少。（通常如果偏业务需求可借由业务常识来定义K）
HAC不直接決定幾個 cluster，而決定你要在樹上切幾刀。


Dimension Reduction
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a769d16e-bff6-4482-81cd-faa3816e4627)

在做 cluster 的時候，就是以偏概全，因為每一個 object都必須要屬於某一個 cluster。

如果你只是把你手上所有的 object分別 assign 到它屬於哪一個 cluster，這樣是以偏概全。

應該要用一個 vector來表示 object，這個 vector裡面的每一個 dimension，代表了某一種特質、某一種 attribute。這件事情就叫做 Distributed 的 representation。

如果原來的 object 是一個非常 high dimension 的東西，比如說 image。

它用它的attribute，把它用它的特質來描述，就會從比較高維的空間變成比較低維的空間，這一件事情就叫做 Dimension Reduction。

實際舉例，其實只需要在 2D 的空間就可以描述這個 3D 的 information，根本不需要把這個問題放到 3D 來解，這樣是把問題複雜化。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b6fa2dce-a640-46d9-bb89-2123acaa429a)


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/884fefff-1e17-4975-8f4c-ac8e12ebcc47)

另一個比較具體的例子， MNIST

在 MNIST 裡面，每一個 input 的 digit都是一個 image，都用 28*28 的 dimension來描述。

但是，實際上多數 28*28 的 dimension 的 vector轉成一個 image看起來都不像是一個數字。所以在這個 28維 * 28維的空間裡面是 digit 的 vector，其實是很少的。所以，其實描述一個 digit，或許根本不需要用到 28*28 維。

比如這邊有一堆 3，然後這一堆 3

如果你是從 pixel 來看待它的話要用 28維 * 28維。然而，實際上這一些 3，只需要用一個維度，就可以來表示。因為這些 3 就只是說把原來的 3 放正，是中間這張 image右轉 10 度轉就變成它，右轉20度就變它，左轉10度變它，左轉20度就變它。

所以，唯一需要記錄的事情只有今天這張 image，它是左轉多少度、右轉多少度，就可以知道說，它在 28 維的空間裡面應該長什麼樣子。

Dimension Reductiong實作
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e0db75d3-4ec5-49b1-89eb-402d9bfcae2e)

找一個 function，function 的 input 是一個 vector, x，它的 output 是另外一個 vector, z。
因為是 Dimension Reduction，所以 output 的這個 vector, z，它的 dimension 要比 input 的 x 還要小。
Feature Selection
最簡單的方法是 Feature Selection，如果把 data 的分布拿出來看一下，本來在二維的平面，但是，你發現都集中在 x2 的 dimension 而已，所以，x1 這個 dimension 沒什麼用，把它拿掉，就等於是做到 Dimension Reduction 這件事。

Principle Component Analysis
另外一個常見的方法叫做Principle Component Analysis (PCA)。

假設這個 function 是一個的 linear function，則 input, x 跟這個 output, z 之間的關係就是一個 linear 的 transform。也就是把這個 x 乘上一個 matrix, W，就能得到它的 output, z。

根據一大堆的 x，我們要把這個 W 找出來。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b3e1a7b8-501f-4fca-9616-c4069d54819f)

PCA 要做的事情就是找這一個 W。

假設現在考慮一個 one dimensional 的 case，只要把 data project 到一維的空間上面，也就是 z 只是一個一維的 vector，也就是一個 scalar，那 W 其實就是一個一個 row ，用 w^1 來表示。

把 x 跟 w 的第一個 row, w^1做 inner product，就能得到一個 scalar, z1。
假設 w^1 的長度是 1，w^1 的 2-norm 是 1。
w 跟 x 是高維空間中的一個點，w^1 是高維空間中的一個 vector，那所謂的 z1 就是 x在 w^1 上面的投影，這個投影的值就是 w^1 跟 x 的 inner product。現在要做的事情就是把一堆 x透過 w^1 把它投影變成 z1。

則現在的目標是希望選一個 w^1經過 projection 以後，得到的這些 z1 的分布是越大越好（也就是說，我們不希望通過這個 projection 以後，所有的點通通擠在一起把本來 data point 和 data point 之間的奇異度拿掉了）。

所以，我們希望找一個 projection 的方向，它可以讓projection 後的 variance 越大越好。

如果我們要用 equation 來表示它的話，就會說現在要去 maximize 的對象是 z1 的 variance，z1 的 variance 就是 summation over 所有的 z1 (z1 - z1\bar) 的平方，z1\bar 就是做 z1 的平均。

在无监督学习中，主成分分析（PCA）是一种常用的数据降维技术。它通过正交变换将可能相关的变量转换为一组线性不相关的变量（主成分），目的是保留数据中的大部分变异性，同时减少维度。

与 K-means 聚类不同，PCA 主要用于探索数据、降低数据维度以及在处理高维数据时发现模式。在 PCA 中我们可以通过计算重构误差来衡量降维的影响。Python 和 scikit-learn 实现 PCA 并应用于 Iris 数据集的示例。

请参考PCA.py 其中一个是使用sklearn toolkit范例，一个是用numpy 来深入了解其算法运作原理。


