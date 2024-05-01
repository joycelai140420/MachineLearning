邻域嵌入（Neighbor Embedding）简介

Neighbor Embedding是一种用于数据降维和可视化的无监督学习技术，它主要用于探索高维数据的内在结构。这类方法的目标是在保持数据中邻近点的局部结构的同时，将高维数据映射到低维空间。最著名的邻域嵌入方法之一是 t-SNE（t-distributed Stochastic Neighbor Embedding）。
向台大老师这个图式如果用PCA就会像直接压扁的，没办法向下图的右示例图，无法把完整的S中的颜色有序排列。就需要Neighbor Embedding技术。

工作原理

保持邻近性：邻域嵌入技术试图确保如果两个数据点在高维空间中彼此接近，它们在低维映射中也应该保持接近。
距离度量：这通常通过优化一个目标函数来实现，该函数度量高维空间和低维空间中的距离差异。

应用场景

数据可视化：在处理复杂的数据集（如遗传数据或图像数据）时，邻域嵌入可以用来降维到2D或3D，使得可以通过可视化来理解数据结构。
特征提取：在预处理阶段，可以使用邻域嵌入作为特征提取的工具，以提高机器学习模型的性能。
异常检测：通过观察低维空间中数据点的分布，可以识别出异常值或离群点。
聚类分析：低维表示可以增强聚类算法的性能，尤其是在原始数据维度非常高时。

优点

有效保留局部结构：相比其他降维技术如PCA，邻域嵌入更擅长于保留数据的局部结构。
适应复杂的流形结构：能够处理位于复杂流形上的数据，这是许多线性降维技术难以实现的。

缺点

计算成本高：尤其是对于大型数据集，算法可能非常耗时。
超参数调整：如 t-SNE 中的困惑度（perplexity）参数需要仔细选择，不同的值可能导致结果差异很大。
难以解释：尽管低维嵌入可用于可视化，但它们通常难以解释，因为转换不如PCA那样直观。

训练时的注意事项

选择合适的超参数：例如，在 t-SNE 中，困惑度（perplexity）和学习率的选择对结果有显著影响。
规模与计算资源：对于非常大的数据集，可能需要使用更高效的算法版本或更多的计算资源。
初始化：特别是在 t-SNE 中，不同的初始化可能会导致结果的显著差异，有时使用 PCA 进行初始化可以提供更稳定的结果。
评估标准：在无监督学习中，没有明确的“准确率”衡量标准，因此需要通过可视化或其他质量评估方法来判断嵌入的质量。

提高训练结果的方法

多次运行与平均：由于邻域嵌入技术，尤其是 t-SNE，可能对初始条件敏感，多次运行并取结果的平均有助于获得更可靠的嵌入。
结合其他技术：有时将邻域嵌入与其他降维技术（如 PCA）结合使用，可以先减少数据的维度再进行邻域嵌入，以提高效率和稳定性。

以下是台大老师Hung-yi Lee授课内容

Dimension Reduction
「非線性」的降維

在高維空間裡面的一個 Manifold

Ex: 地球

表面就是一個 Manifold
塞到了一個三維的空間裡面
Euclidean distance (歐式幾何) 只有在很近的距離的情況下才會成立
Ex: 附圖之S形空間

藍色區塊的點距離近 Rightarrow​ 他們比較像
距離比較遠(ex: 藍色跟紅色) Rightarrow 無法直接以 Euclidean distance 計算相似度

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f74e0777-7dcb-4241-96d6-8a5fd2dc8363)

Manifold Learning
把 S 形的這塊東西展開

把塞在高維空間裡面的低維空間「攤平」，也就是降維

Pros

可以用 Euclidean distance 來計算點和點之間的距離
對 clustering 有幫助
對 supervised learning 也會有幫助

Locally Linear Embedding (LLE)
Setting
本來有某一個點，叫做 x_i
然後選出這個 x_i 的 k 個 neighbors
假設其中一個叫做 x_j ，w_{ij} 代表 x_i 和 x_j 的關係
表示所有的 k 個 neighbors Neighbor 之線性組合要跟 x_i 越像越好
Minimize sum_i |x^i - sum_j w_{ij} x^j \_2

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b74d8e6a-2516-4a91-9aa3-1679f5edb0ef)

Dimension Reduction
將所有的 x_i​ 跟 x_j​ 轉成 z_i​ 和 z_j​ ，而中間的關係 w_{ij}​ 是不變的
首先 w_{ij} 在原來的 space 上面找完以後，就 fix 住
沒有一個明確的 function 說怎麼做 dimension reduction
憑空找出來降維後的 z_i 跟 z_j ，可能原本100維(x)，降到2維(z)
Minimize sum_i |z^i - sum_j w_{ij} z^j |_2

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b9d1623b-ca19-492e-9f6c-2959aa7fa802)

Something about Numbers of Neighbors (k)
neighbor 選的數目要剛剛好才會得到好的結果
Reference paper: “Think Globally, Fit Locally”
k 太小，就不太robust，表現不太好
k 太大，會考慮到一些距離很遠的點，這些點被 transform 以後，relation 沒有辦法 keep 住

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d7634f8f-b3ed-4118-be0d-f991dd86e1ca)


请参考Locally Linear Embedding_LLE.py

Laplacian Eigenmap
考慮到先前提過的 Smoothness assumption

只算它的 Euclidean distance 來比較點跟點之間的距離是不足夠的

要看在這個 High density 的 region 之間的 distance

有 high density 的 connection 才是真正的接近
可以用 graph 描述
Graph Construction

計算 data point 兩兩之間的相似度，超過一個 thereshold 就 connect 起來

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b1f013f6-3b72-41c1-b848-9991db9662f5)

考慮到先前提過的 Smoothness assumption

只算它的 Euclidean distance 來比較點跟點之間的距離是不足夠的

要看在這個 High density 的 region 之間的 distance

有 high density 的 connection 才是真正的接近
可以用 graph 描述
Graph Construction

計算 data point 兩兩之間的相似度，超過一個 thereshold 就 connect 起來


