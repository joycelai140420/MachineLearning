无监督学习 - 词嵌入（Word Embedding）简介

词嵌入是自然语言处理（NLP）中的一种技术，旨在将词汇表中的词语转换为密集的向量表示。这些向量通常由浮点数构成，能够捕捉单词之间的语义和句法关系。词嵌入通常是通过无监督学习方法从大量文本数据中学习得到的。

主要方法

Word2Vec：Google 的一种实现，使用两种架构，CBOW（Continuous Bag of Words）和Skip-gram，通过预测上下文或中心词来学习词向量。
GloVe（Global Vectors for Word Representation）：Stanford 的一种实现，它基于整个语料库中的词汇共现统计信息，通过矩阵分解技术来学习词向量。
FastText：由 Facebook 开发，类似于 Word2Vec，但它在学习词向量的同时，也考虑了单词的内部结构（如前缀和后缀）。

应用场景

文本相似性检测：使用词向量计算文本之间的相似度。
情感分析：基于词向量构建模型，判断文本的情感倾向。
机器翻译：利用词向量改进翻译质量。
信息检索：通过词向量提高搜索结果的相关性。

优点

降低维度：词嵌入将高维的稀疏表示（如独热编码）转换为低维的密集向量。
捕捉语义：能够有效地捕捉单词之间的语义关系，如同义词、反义词等。
通用性：学习到的词向量可以跨多个任务和领域使用。

缺点

静态表示：传统的词嵌入方法为每个词提供一个固定的向量，不能很好地处理语境中的多义性。
需要大量文本：为了学习有效的词向量，需要大量的文本数据。
更新困难：随着语言的发展，模型需要不断更新以包括新词汇和表达。

提升训练结果的策略

数据预处理：彻底的文本清洗和标准化可以提高模型的质量。
选择合适的架构：根据具体任务选择最合适的词嵌入模型，如选择 Word2Vec 的 Skip-gram 模型来处理小型数据集。
调整超参数：优化学习率、窗口大小、向量维度等超参数。
负采样：在训练过程中使用负采样可以提高训练速度和效果。
使用子词信息：FastText 通过使用词根信息来提升对未知词汇的处理能力


测试集上提升准确率的方法

评估指标选择：选择合适的评估指标，如使用余弦相似度来衡量词向量之间的相似性。
集成学习：将不同的词嵌入模型结果进行集成，以提高测试的稳健性。
细化调试：通过错误分析来指导后续的模型调整。


增加模型准确率的其他动作

持续学习：定期在新收集的语料上重新训练或微调模型，以适应语言的变化。
跨语言训练：在多语言数据上训练模型，提高模型的泛化能力。
语境相关词嵌入：使用像 BERT 或 ELMo 这样的上下文词嵌入技术，以更好地处理词语的多义性和语境变化。

接下來以下內容截取臺大Hung-yi Lee 授課內容，我会根据内容来实现几个范例

Word Embedding

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e0f3b467-47b5-4554-a30d-71b8f36b7ab6)

Word Embedding 是 Dimension Reduction的一種。主要的過程便是用Vector來表示Word。主要有兩種方法：

1-of-N encoding

第一種叫做 1-of-N encoding，每一個 word，用一個vector來表示 這個 vector 的 dimension，是所有的 word 數目。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2b44d085-b388-40c9-8c9a-6b10d47b0f6c)

假設這個世界上有十萬個 word，那 1-of-N encoding 的 dimension就是十萬維。每一個 word，對應到其中一維。

所以，對apple這個word的vector來說 ，它的第一維是 1，其他都是 0。bag 就是第二維是 1，cat 就是第三維是 1，以此類推...

缺點：如果用這種方式來描述一個 word，vector 會比較不 informative。 由於每一個 word，它的 vector 都沒什麼關聯，所以從這個 vector 裡面，沒有辦法得到其他的資訊。（比如說，cat 跟 dog都是動物這件事，你便沒辦法知道） 因此之後就有另一個方法就做Word Class。

A clustering method: Word Class
Word Class，就是把有同樣性質的 word，cluster 成一群一群的，然後用那一個 word 所屬的 class 來表示這個 word。這個就是 Dimension Reduction 裡面 clustering 的概念。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/85952ec6-f202-4f10-9a50-76019d37e386)

比如說，dog, cat 跟 bird，它們都是class 1，ran, jumped, walk 是 class 2，flower, tree, apple 是class 3，等等...

但是，光用class 其實是不夠的。光做 clustering 的話，一些 information（比如說，動物的 class與植物的 class，都屬於生物）便沒有辦法被呈現出來。class與class之間的關聯無法被顯現。

因此，我們需要的是 Word Embedding。

Word Embedding
Word Embedding 就是把每一個 word都 project 到一個 high dimensional 的 space 上面。

雖然說，這個 space 是 high dimensional的，但是它的維度其實遠比 1-of-N encoding 的 dimension 還要低。

實際上，對1-of-N encoding來說，英文有 10 萬詞彙，它就是 10 萬維但如果是用 Word Embedding ，通常是 50 維、100維的 space就可以了。或者可以說從 1-of-N encoding 到 Word Embedding，就是一個
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ae22b721-b173-44b6-96eb-3359f0f03ac4)

以上面這張圖word embedding 的部分為例，我們可以看到的結果是類似 semantic，類似語意的詞彙在這個圖上是比較接近的。而且在這個 Word Embedding 的 space 裡面，每一個 dimension，可能都有它特別的含意。比如說上圖的橫軸的dimension可能就代表了生物和其他的非生物之間的差別，而縱軸這個 dimension 可能就代表了跟動作與靜止的關聯。

那怎麼做 Word Embedding呢？

Word Embedding 是一個 unsupervised 的 approach。

要讓 machine知道每一個詞彙的含義是什麼，需要讓 machine 閱讀大量的文章，來確定embedding後的feature vector 應該長什麼樣子。

這是一個 unsupervised 的 problem，實際上做的事情就是learn 一個 Neural Network， input 是詞彙，output 則是那一個詞彙所對應的 Word Embedding裡的 vector。

然而我們手上有的 train data只是一大堆的文字，所以我們只有Input，但是我們沒有 output，我們不知道每一個 Word Embedding 應該長什麼樣子。所以對於我們要找的 NN function，我們只有單項的輸入，不知道輸出。

因此，這才是一個 unsupervised learning 的問題。

Can Auto-encoder solve word embedding?

前面的章節有講解過一個 deep learning base 的 Dimension Reduction，叫做 Auto-encoder。

實際上需要的做法是 learn 一個 network，讓它輸入等於輸出，再將某一個 hidden layer 拿出來，作為 Dimension Reduction 的結果。

然而在這個地方，並沒有辦法用 Auto-encoder 來解

舉例來說：

如果用 1-of-N encoding 當作它的 input，對 1-of-N encoding 來說，每一個詞彙都是 independent 的。這樣子的 vector 做 Auto-encoder，沒有辦法 learn 出任何 informative 的 information。就算用character 的 n-gram 來描述一個 word，也只可以抓到一些字首、字根的含義。所以基本上，現在大家並不是這麼做。

在 Word Embedding 這個 task 裡面，沒有辦法使用Auto-encoder

Word Embedding Implementation

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/fcf3770d-f861-43e7-b900-b008bbcf284a)

Word Embedding基本的精神：每一個詞彙的含義，可以根據它的上下文來得到。

舉例來說：

假設機器讀了一段文字：『馬英九520宣誓就職』，它也讀了另外一段新聞：『蔡英文520宣誓就職』。

對機器來說，雖然它不知道馬英九指的是什麼，也不知道蔡英文指的是什麼，但是馬英九後面有接520宣誓就職，蔡英文的後面也有接520宣誓就職。對機器來說，只要它讀了大量的文章並且發現馬英九跟蔡英文前後都有類似的 context機器就可以推論說：「蔡英文跟馬英九代表了某種有關係的 object」。它可能也不知道他們是人，但它知道，蔡英文跟馬英九這兩個詞彙代表了，同樣地位的某種東西。

那怎麼來實做這一件事呢？
怎麼用這個基本的精神來找出 Word Embedding 的 vector，有兩個不同體系的作法
    Count based 的方法
    Prediction based 的方法
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b703a5d4-1439-49d2-a9c0-0a321eaf089a)

概念：現在有兩個詞彙，w_i, w_j，它們有時候會在同一個文章中出現（co-occur）。我們用 V(w_i) 來代表w_i 的 word vector，我們用 V(w_j) 來代表w_j 的 word vector。如果 w_i 跟 w_j常常一起出現的話，V(wi) 跟 V(wj) 就會比較接近。

這種方法，有一個很代表性的例子，叫做 Glove vector。

假設我們知道：wi 的 word vector 是 V(wi), wj 的 word vector 是 V(wj)。

計算它們的 inner product（假設為N_ij） ，則為 wi 跟 wj co-occur 在同樣的 document 裡面的次數。

Count-based的目標就是為 wi 找一組 vector，也為 wj 找一個組 vector，co-occur的次數跟inner product的直越接近越好。

這個概念與 LSA 和 matrix factorization 的概念類似。








