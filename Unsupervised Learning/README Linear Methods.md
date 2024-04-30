
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

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2c42f2ee-9749-44c6-bb99-258eaa7491ce)

