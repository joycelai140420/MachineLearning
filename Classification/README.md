这个章节是作为学习机器学习的第一堂课，我们先介绍一下Machine Learning 比较常态Components，整個過程大致可以分成幾個核心組件，每個組件都扮演著重要的角色。我將用一個簡單的例子來說明每個組件：

數據(Data)：
      這是機器學習系統的基石。數據可以來自多種來源，例如圖片、文字、數字等。
      
      例子：假設我們要建立一個機器學習模型來辨識郵件是否為垃圾郵件，這裡的數據就是郵件的內容和相關標籤（垃圾郵件或非垃圾郵件）。

特徵(Features)：
      
      特徵是從原始數據中提取的重要信息，用於訓練模型。選擇好的特徵對於建立一個有效的模型非常關鍵。
      
      例子：在垃圾郵件辨識的例子中，特徵可能包括郵件中的關鍵詞（如“免費”，“促銷”），發件人的信譽等。
      
模型(Model)：
      
      模型是根據特徵來進行學習和預測的算法。有許多類型的模型，例如決策樹、支持向量機（SVM）、神經網絡等。
      
      例子：選擇一個適合處理文本數據的模型，比如邏輯回歸，來進行是否為垃圾郵件的二元分類。
學習算法(Learning Algorithm)：
      
      學習算法是指導模型從數據中學習的方法。這可能包括監督學習、非監督學習、強化學習等。

      例子：在我們的垃圾郵件分類模型中，使用監督學習，因為我們有郵件的標籤作為訓練指導。

評估(Evaluation)：
      
      評估是用來測試模型效能的過程。通常會用一些沒有用於訓練的數據來進行評估，以確保模型的泛化能力。

      例子：我們可以使用精確度（accuracy）和召回率（recall）來評估垃圾郵件分類器的性能。

優化(Optimization)：
      
      優化是改進模型性能的過程，可以通過調整模型參數（如學習率）、增加更多訓練數據等方式進行。
      例子：如果發現模型對於某類郵件的分類效果不佳，可能需要重新調整特徵選擇或者優化模型的參數。

图片截取台大Hsuan-Tien Lin老师课程
![1715325848466](https://github.com/joycelai140420/MachineLearning/assets/167413809/fcaa2de3-c6af-416e-a7e9-97c14bb9ed18)

其中提到了Hypothesis Set。

在機器學習中，特別是在談論特定算法如感知器（Perceptron）時，「假設集（Hypothesis Set）」確實是一個重要的概念。

假設集（Hypothesis Set）

假設集是所有可能的模型（或函數）的集合，這些模型被考慮用於從特徵預測輸出。在機器學習中，學習的目標是在假設集中選擇最佳的模型，使其能夠最好地解釋數據並進行預測。至于Hypothesis Set是怎麼影響模型的好壞，可以參考MachineLearning/Collection of tips
/Loss and Training set tips.md ，這邊暫時不提。

感知器的假設集

在感知器算法中，假設集通常包含所有可以通過一個線性函數來表示的決策界面。而感知器模型本質上是一個二元分類器，它通過一個線性方程來劃分數據點：

![1715326439719](https://github.com/joycelai140420/MachineLearning/assets/167413809/5e1d8b6f-437f-4743-ba57-bb0bc522d164)

其中 𝑤 是權重向量，𝑥 是特徵向量，𝑏 是偏置項，sign函數根據線性方程的結果返回 +1 或 -1。

簡單例子

假設我們正在建立一個模型來判斷一個學生是否會通過最終考試，基於他們的學習時間和前次考試分數。在這裡：

特徵 𝑥：學習時間和前次考試分數。

假設 ℎ：一個線性函數 ℎ(𝑥)=𝑤1×學習時間+𝑤2×考試分數+𝑏。

模型預測：如果 ℎ(𝑥)≥0，預測學生會通過考試；否則，不會。

在這個例子中，感知器的假設集包含了所有可能的線性決策界面，而學習算法的目標是找到最適合數據的權重 𝑤 和偏置 𝑏。



又或者可以參考下面台大Hsuan-Tien Lin老师课程圖示，這範例是衝從顧客特徵的訊息中來決定是否要發信用卡。

會有一個閾值來決定是否放貸，如果這個人的W權重xXi大於等於某個閾值，就發信用卡，反之就不發信用卡。

也就是說把使用者資料拿來， 透過 W 來做一個加權，然後取一個總分， 然後看看這個總分有沒有超出我們的門檻值，超出來我們就給正1， 沒有超出來我們就給，沒有就給負1，  那你說那如果剛好在門檻上面怎麼樣？ 通常這種事情很少發生，我們可以想像說我們就當作 就就就是一個這個這個特定的狀況，我們可以暫時不管它 ，

小 h 是什麼？小 h 是我們可能的公式。 這個小 h 跟什麼東西有關，跟 W 有關，跟我們選的門檻有關，所以我們可以 不同的 W ，不同的門檻，就造出不同的 h ， 好，那這樣的 h，在這個歷史上我們把它叫做 perceptron， perceptron 這個字字面上翻譯叫做感知器。

 注意到 hypothesis 意思是說電腦最後會猜測，猜測說是不是這是一個可能的公式的長相，那這個 W 通常我們會叫做 weight，就是這個權重 我們的每一個這個維度的權重。

![1715327149408](https://github.com/joycelai140420/MachineLearning/assets/167413809/c6fcc262-477e-4221-a44d-920fb9896294)

那麼知道Machine Learning 比较常态Components，因為線性的有時候常難以分類，那麼接下來就是介紹一個分類模型Logistic Regression（逻辑回归）

Logistic Regression（逻辑回归）简介

逻辑回归是一种广泛用于分类问题的统计模型，尤其是在二分类问题中非常流行。它通过使用逻辑函数（Logistic function），也称为Sigmoid函数，将线性回归的输出映射到0和1之间，从而预测分类的概率。

算法原理
逻辑回归的基本形式可以表示为： 𝑦^=𝜎(𝑤𝑇𝑥+𝑏) 其中：
•	𝑥 是特征向量
•	𝑤 是权重向量
•	𝑏 是偏差（bias）
•	𝜎(𝑧) 是Sigmoid函数，定义为 𝜎(𝑧)=1/1+𝑒−𝑧

Sigmoid函数的输出是一个介于0和1之间的概率值，表示样本属于正类的概率。

应用
二分类问题：如垃圾邮件检测、疾病诊断、客户流失预测等。
多分类问题：通过一对多（OvR）或多对多（OvO）策略，可以扩展逻辑回归到多类分类问题。
概率评估：逻辑回归不仅给出分类结果，还能提供决策的概率基础，这对于需要评估风险的应用非常有用。

优点
1.	模型简单：逻辑回归模型形式简单，易于理解和实现。
2.	计算效率高：相比于复杂的模型，逻辑回归的训练和预测速度较快。
3.	输出概率：能够输出预测的概率，这对于需要概率解释的应用很有帮助。

缺点
1.	表达能力有限：逻辑回归假设数据是线性可分的，对于复杂的模式或非线性关系可能表现不佳。因為他就是一個直線，如下圖，但可以使用Feature Transformation例如cascading Logistic Regression models
   
   ![1713792877050](https://github.com/joycelai140420/MachineLearning/assets/167413809/f5c38891-76ef-44a1-a360-231e60e80a0a)
![5b34bb09d1e8f00c579d0131aba2ff5](https://github.com/joycelai140420/MachineLearning/assets/167413809/3818c7cc-4db8-497b-a30f-e94561aca279)



2.	高度依赖数据表示：对特征工程高度依赖，数据预处理和特征选择的质量直接影响模型性能。

相关内容详解请参考台大Hung-yi Lee老师的课程。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c91df5d0-936c-4cfc-bd3d-13bcf4d4b71b)

实现逻辑回归的代码请参考：Logistic Regression.py

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e1e6d992-c2eb-4116-b6a8-275b8d6a5173)

Logistic Regression：有通過 $sigmoid function$，output 的值介於 0~1
Linear Regression：單純將 $feature*w+b$，output 可以是任何值

那么这边为什么推荐用cross entropy是因为参数update的时候变化量越大，步伐就可以跨越大，越快跑到目标点，square error 是红色的线，比较平滑，所以参数update的时候变化量越小，步伐就可以跨越小，越慢跑到目标点。
![1713779854769](https://github.com/joycelai140420/MachineLearning/assets/167413809/740169bc-bc38-4652-b9e3-fdbc12c58f76)

Probabilistic Generative Models跟Logistic Regression不同是，Logistic Regression（Discriminative）是不做任何假设，但是Probabilistic Generative Models（Gernerative)对probability distribution是有假设前提，例如假设是伯努利分布等等。从文献上很多人会说Discriminative表现比较好，那么什么时候用Gernerative会比较好呢？参考台大Hung-yi Lee老师的课程说明。

何時 Generative model 的表現較比 Discriminative model 好

資料量大小：

Discriminative model 因爲不做任何假設，故 performance 受資料影響很大 Generative model 會做假設（如同自行腦補），資料量很少時，較有優勢
資料量小：Discriminative model 誤差較大，Generative model 表現可能較好 資料量大：Discriminative model 誤差較小，表現較有可能優於 Generative model

Noise 存在：

資料有 noise 時，因為 label 本身就有些問題，故一些假設可能可以把有問題的 data 忽略掉 Generative model 的表現可能較 Discriminative 好

分割資料來源：

Discriminative model 直接假設一個 posterior probability Generative model 可將 formulation 拆成 prior 跟 class-dependent 的 probability 兩項 而這兩項可以來自不同的資料來源

舉例： 語音辨識使用 NN，是 discriminative 的方法； 但是整個語音辨識系統，是 generative 的 system。

prior 的部分使用文字的 data 處理，class-dependent 的部分，需要聲音和文字的配合。因为有时候我们不知道别人会说什么，所以有时候需要有概率的脑补对方会说什么概率。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/25b4f578-5331-4924-a858-3ec3501078f0)

參考：Probabilistic Generative Models of Bernoulli Distribution.py
不過要特別留意代碼裡的Likelihood，會跟老師說的不一樣，在讨论概率生成模型中的 likelihoods 时，我们遇到了一个常见的误解点。首先，我们需要区分两个不同的概念：“似然”(likelihood) 和 “损失函数” 中用于训练模型的项，如交叉熵损失函数中的 𝑦^ln𝑦项。

似然 (Likelihood)

在统计学和概率生成模型中，似然函数是关于模型参数的函数，给定观测数据时，它表示这些参数的“合理程度”或“可能性”。其中 θ 是该特征为1的概率。这就是为什么在我们的代码中用样本均值来估计 θ，因为这是最大似然估计。

交叉熵损失（cross Entropy)

在分类任务中，尤其是使用逻辑回归时，常用的损失函数是交叉熵损失，它衡量的是模型预测的概率分布 𝑦^与真实标签 y 之间的差异。这里的 y^是模型预测给定输入x 属于正类的概率。这个损失函数用于参数优化，确保模型预测尽可能接近真实标签。

区别

在生成模型的设置中，likelihoods 数组中的每个值 θ 直接表示给定类别下特定特征为1的概率。这是模型的一个内部参数，用于描述数据生成过程。而 𝑦^ln𝑦是评估模型输出与真实数据之间差异的一个工具，用于学习模型参数。

在生成模型中，我们通常不直接用 𝑦^ln𝑦这样的表达式去学习𝜃，因为我们的目标是直接模拟数据的潜在分布，而不是最小化预测误差。因此，通过估计每个类别下各特征为1的概率（即伯努利参数），我们可以使用这些概率来计算新样本的类别概率，这通常通过应用贝叶斯规则来完成。希望这解释清楚了为什么在生成模型中，我们关注的是直接估计数据分布的参数，而不是使用类似于逻辑回归中的交叉熵损失表达式
