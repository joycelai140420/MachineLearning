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

请参考Laplacian Eigenmap.py范例

Laplacian Eigenmap改善了数据表示的几个关键方面：
1. 保持局部邻近性：意味着如果两个数据点在原始高维空间中是邻近的，它们在降维后的低维空间中也应保持接近。
2. 解决非线性降维问题：与传统的线性降维技术（如 PCA）相比，Laplacian Eigenmap 能够处理非线性结构的数据。
3. 增强数据可视化：通过将高维数据有效地映射到二维或三维空间，Laplacian Eigenmap 改善了数据的可视化，使研究人员能够直观地观察和分析数据的结构和模式
4. 聚类支持：由于它保持局部邻近性，Laplacian Eigenmap 通常可以增强聚类算法的性能。在降维后的空间中，具有相似特性的数据点会自然聚集在一起，从而简化了聚类过程并可能提高其准确性。
5. 数据预处理：在复杂的机器学习或模式识别任务中，Laplacian Eigenmap 可以作为预处理步骤来改善后续算法的性能。通过降维，可以减少计算复杂性和避免“维数的诅咒”，同时保留对任务有用的信息。

t-SNE (T-distributed Stochastic Neighbor Embedding)
Problems above
先前只假設相近的點應該要是接近的，但沒有假設不相近的點不要接近、要分開
確實會把同個 class 的點都聚集在一起，但也會把不同的class混雜
Example: MINIST (手寫辨識)、 COIL-20 (Image Corpus)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e6763770-36b0-4b5a-853e-359b8c7fd73b)

一般t-SNE 运算量大，通常可以先用PCA降维到50维，再用t-SNE 降维从50维到2维。

t-SNE（T-distributed Stochastic Neighbor Embedding）是一种非常流行的机器学习算法，专门用于高维数据的可视化。它由 Laurens van der Maaten 和 Geoffrey Hinton 在 2008 年提出。t-SNE 通过一种特别有效的方式来捕捉高维数据的局部结构，并且能在低维空间中（通常是二维或三维）以一种可视化的形式展示这些结构。

t-SNE 的工作原理：
1.相似性计算：t-SNE 首先计算高维空间中每对数据点之间的相似性，这通常通过高斯分布（正态分布）来实现。对于每个点 𝑥𝑖 ，计算它与另一个点𝑥𝑗 的条件概率 𝑝𝑗∣𝑖 ，这个概率反映了选择𝑥𝑗作为𝑥𝑖的邻居的相对概率。
2.对称相似性：为了简化问题，t-SNE 进一步计算点 𝑖 和 𝑗 的联合概率 𝑝𝑖𝑗，使其对称化。
3.低维映射与相似性：在低维空间中，t-SNE 同样计算每对点之间的相似性，但使用了 t 分布（具有更重的尾部）来计算点 𝑦𝑖和 𝑦𝑗之间的相似性 𝑞𝑖𝑗4.KL 散度最小化：目标是让低维空间中的 𝑞𝑖𝑗尽可能接近高维空间中的 𝑝𝑖𝑗 。


t-SNE 算法中将不相近的点保持距离的关键步骤在于它如何在低维空间中模拟高维空间中的相似性和不相似性。

其中，t-SNE 使用 t-分布t分布比正态分布有更重的尾部，这意味着它对较远的点赋予更大的概率。这种分布选择的直接后果是，低维空间中距离稍远的点之间的相似性𝑞𝑖𝑗会被相对放大，而这在高维空间中对应的相似性 𝑝𝑖𝑗较小。这有助于在低维表示中拉开这些点之间的距离，即使它们在原始高维空间并不接近。

t-SNE 的核心是最小化高维和低维空间中相似性分布之间的 Kullback-Leibler (KL) 散度。KL 散度衡量两个概率分布之间的差异，其特点是它对低维空间中不正确建模远距离的点（即高维中不相似但低维中错误地近似）非常敏感。在 t-SNE 的优化过程中，如果两个点在高维空间中是不相似的（即𝑝𝑖𝑗很小），而在低维空间中它们却错误地靠得很近（使得𝑞𝑖𝑗相对较大），这会导致较大的 KL 散度贡献。因此，优化过程会尝试调整这些点的位置，以增加它们在低维空间中的距离，从而减少这种不一致性。

请参考t-SNE.py

