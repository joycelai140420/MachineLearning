Generation 

在无监督学习领域，生成模型（Generative Models）是一类旨在学习如何生成数据的真实分布的模型。这类模型尝试捕获数据的底层概率结构，以便能够生成新的、与原始数据统计上不可区分的实例。


常见的生成模型包括：
1.生成对抗网络（GANs，Generative Adversarial Networks）
2.变分自编码器（VAEs，Variational Autoencoders）
3.玻尔兹曼机（Boltzmann Machines）和限制玻尔兹曼机（Restricted Boltzmann Machines, RBMs）
4.PixelRNNs/PixelCNNs

优点：

新数据生成：能够生成新的数据实例，这在数据增强、艺术创作和模拟实验中非常有用。
数据理解：通过模拟数据生成过程，帮助更深入地理解数据的内在特征和结构。
少量数据学习：生成模型可以在少量数据上训练，并能够推广到新实例的生成，这在数据稀缺的场景中尤其重要。
特征表示：可以学习到强大的、有意义的数据表示，支持其他机器学习任务，如分类和回归。

缺点：
训练难度：许多生成模型，尤其是 GANs，在训练上有稳定性和收敛性的问题。
计算成本：生成模型通常计算成本高，需要大量的计算资源，尤其是在处理大规模数据集时。
模式坍塌（对于 GANs）：在训练生成对抗网络时，可能遇到模式坍塌的问题，即生成器开始生成极少数的样本类型。

应用：

图像和视频生成：用于艺术创作、电影制作和视频游戏中的自动内容生成。
数据增强：在医学图像处理、自动驾驶车辆和其他需要大量训练数据的领域中增强现有数据集。
无监督特征学习：通过训练生成模型来学习有用的数据表示，进而改善监督学习任务的性能。
模拟和预测：在金融、气象和生物学等领域用于模拟可能的未来事件和情况。


以下是台大老师Hung-yi Lee授课内容

Generative Models

Component-by-component (PixelRNN)
Variational Auto-encoder (VAE)
Generative Adversarial Network (GAN)

上述方法都很新，皆是都是近幾年提出的。

Component-by-component

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3ca391a2-96cc-4eaa-b460-0d81cb0dc6c5)

將圖片攤平，用 RNN 以之前的 pixel (RGB三圍向量)去 predict 下一個 pixel，把整張圖畫出來 (unsupervised)

不只用在圖片，還可以用在語音，像是WaveNet。也可以用在影片：給定一段 video，讓他predict 接下來會發生甚麼事。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/21909bff-a7e2-43b6-aa65-1cd5ea83ba8a)
自己去创作宝可梦
在这里解释一下1-of-N 编码（或称为 one-hot 编码），比如性别（男/女）、城市（纽约、伦敦、北京）等。为了将这类类别数据转换为机器学习算法可以处理的格式，我们使用一种方法将类别变量转换为数值变量。1-of-N 编码通过为每个类别分配一个独特的二进制向量来实现。在这个向量中，只有一个元素是1，其余都是0。向量的长度等于可能类别的数量。通常，编码的顺序会根据类别在数据中出现的顺序或者按字母、数值等自然顺序进行排序。这种编码顺序在大多数情况下是由使用的编程库或工具在内部决定的，但也可以人为指定。
大部分数据处理库，如 pandas 和 scikit-learn，会自动根据数据中出现的顺序或者字母顺序进行编码。例如，在 pandas 中使用 pd.get_dummies() 函数时，列的排序默认是按照字母顺序的，这意味着如果类别标签是字符串，它们将按字典顺序被编码。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9a9bd00a-cb1e-446e-b5e4-cf9779cc5e61)
要對產生出的 Image 做 Evaluation 是很困難的，且通常Unsupervised Learning是无法给出好坏。因为是主张创造出来。

 Auto-encoder
 ![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c52057e5-db12-40c8-bf33-a95470339831)
Input: image => Encode: low dimension code => Decode: Image
讓 Input & Output兩張圖越接近越好

Given random code => Decode: Image?
效果差

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2f5a3284-b43d-4732-8b73-7c3231ca0549)

VAE比起 Auto-encoder，加了小 trick：不直接 output code，而是先 output 兩個 vector，再與random出來的 Vector 做如圖的運算，當作 code。

目的是：minimize reconstruction error，雖然結果沒有 PixelRNN 清晰，但 code 的每一個 dimension 代表特定意思。这也可以用來寫詩，將 IO 從 Image 換成 Sentence

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1511dac3-48df-476e-96de-b83da1a4a239)
舉例：以 Pokemon Image 生成為例：
將其中八維固定，以其他兩維 Random 變化作圖：
知道code 的每一个dimension背后的意思是什么，就可以控制大概输出的方向。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/60b4fdeb-6bb1-4fb0-ba52-4396085aea84)
可以看出這兩個維度大概表達的意思，分別是腳以及尾巴 (雖然不是很明顯)
可以透過調整得到一個看起來最 OK 的 Pokemon

