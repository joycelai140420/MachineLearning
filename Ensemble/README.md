Ensemble Learning

集成学习（Ensemble Learning） 是一种机器学习范式，它通过结合多个模型的预测来提高整体性能。让我们深入探讨一下。

核心思想：多个模型的集合通常比单一模型更能准确地预测未知数据。

优点：

    提高预测性能：集成学习通常比单一模型更准确。

    减少过拟合：通过结合多个模型，可以减少过拟合的风险。

    对噪声数据具有鲁棒性：集成方法对噪声数据的影响较小。

缺点：

    计算开销：训练多个模型需要更多的计算资源。

    可解释性：集成模型通常比单一模型更难解释。

    数据不平衡：在某些情况下，集成方法可能对数据不平衡敏感。

以下就是来自于台大老师Hung-yi Lee的课程内容

知道Ensemble Learning是透过一个多个模型的方式，来提高整体性能。

首先先講 Bagging 這個方法 注意一下等一下除了會講 Bagging 之外還會講 Boosting，Bagging 和 Boosting 使用的場合是不太一樣的

![1714813494331](https://github.com/joycelai140420/MachineLearning/assets/167413809/59cbddd6-0035-4078-9b0a-9c1fb76c7809)


如果有一個很簡單的 Model 我們會有很大的 Bias、比較小的 Variance

如果我們有複雜的 Model 可能是小的 Bias、大的 Variance

在這兩者的組合下，我們會看到我們的 Error rate 隨著 Model 的複雜度增加逐漸下降再逐漸上升

我们可以把不同的模型统统组合起来。

![1714813639873](https://github.com/joycelai140420/MachineLearning/assets/167413809/b16217ac-c012-4b38-aac2-561704f7ecdc)

那 Bagging 其實就是要體現這件事情，Bagging 要做的事情就是 雖然我們不可能蒐集非常多不同的data，但是我們可以創造出不同的 data set 再用不同的 data set 各自訓練一個複雜的 Model，雖然每一個 Model 獨自拿出來看可能 Variance 很大 ，把不同的 Variance 很大的 Model 集合起來以後，他的 Variance 就不會那麼大 他的 Bias 會是小的。

怎麼自己製造不同的 data 呢 ？
假設現在有 N 筆 Training Data，對這 N 筆 Training Data 做 Sampling 從這 N 筆 Training Data 裡面每次取 N' 筆 data，組成一個新的 Data Set 通常在做 Sampling 的時候會做 replacement，抽出一筆 data 以後會再把它放到 pool 裡面去 ，那所以通常 N' 可以設成 N，所以把 N' 設成 N 從 N 這個 Data Set 裡面 做 N 次的 Sample with replacement，得到的 Data Set 跟原來的這 N 筆 data 並不會一樣，因為你可能會反覆抽到同一個 example 。

總之我們就用 sample 的方法建出好幾個 Data Set，每一個 Data Set 都有 N' 筆 Data 每一個 Data Set 裡面的 Data 都是不一樣的，接下來 你再用一個複雜的模型 ，去對這四個 Data Set 做 Learning，就找出了四個 function 。

![1714813931076](https://github.com/joycelai140420/MachineLearning/assets/167413809/f97821b0-d9fe-4fe7-a641-692ee3a1191b)

接下來在 testing 的時候，就把一筆 testing data 丟到這四個 function 裡面 再把得出來的結果作平均，或者是作 Voting 通常就會比只有一個 function 的時候performance 還要好 ，performance 還要好是指說你的 Variance 會比較小，所以你得到的結果會是比較 robust 的 比較不容易 Overfitting。
如果做的是 regression 方法的時候 ，你可能會用 average 的方法來把四個不同 function 的結果組合起來。如果是分類問題的話可能會用 Voting 的方法把四個結果組合起來,就看這四個 function 裡面哪一個類別 最多 classifier 投票給他，就選那個 class 當作 Model 的 output。

注意一下，甚麼時候做 Bagging 當你的 model 很複雜的時候、擔心它 Overfitting 的時候，才做 Bagging 做 Bagging 的目的是為了要減低 Variance。就是 Model 很容易 Overfitting 的時候要做 Bagging， 所以 Decision Tree 很需要做 Bagging。例如：Decision Tree就是很容易Overfitting 。

![1714814241107](https://github.com/joycelai140420/MachineLearning/assets/167413809/371dbd7b-87d0-43f6-8f6f-de306c44b2f7)

現在假設每個 object，有兩個 feature x1 跟 x2，Decision Tree 就是根據 Training data 建出一棵樹，這棵樹告訴我們，如果輸入的 object x1 < 0.5 就是 yes x1 > 0.5 就是 no，所以就是在 x1 = 0.5 的地方切一刀， 以左就走到左邊這條路上去， 往右就走到右邊這條路上去。接下來再看 x2 < 0.3 的時候 那就說是 Class 1，是藍色，x2 > 0.3 的時候就說是 Class 2，就紅色， 那如果在右邊呢？右邊如果 x2 < 0.7 的時候就塗紅色 x2 > 0.7 的時候就塗藍色。它只看一個 dimension ，其實可以同時看兩個 dimension，也其實可以問更複雜的問題。所以做 Decision Tree 的時候會有很多需要注意的地方， 舉例來說，在每個節點做多少分支？要用甚麼樣的 criterion 來做分支？ 要甚麼時候停止分支？
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/08cc2381-fc8f-48ef-a717-a8db7246ff2d)

舉一個 Decision Tree 的例子吧 ！這個分類的問題是說輸入的 feature 是二維。這個紅色的部分是屬於 Class 1 在藍色的部分是屬於另外一個 Class，而Class 1 分佈的樣子正好就跟初音是一樣的，Class 2 分佈的樣子，就是背景。训练的data，如下网址：http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/theano/miku
第一列是x,第二列式y，第三列是labels

![1714815047432](https://github.com/joycelai140420/MachineLearning/assets/167413809/90c97e4b-5499-4f5d-849c-69970374500d)

当 Decision Tree 的深度是 5、10、15、20的图形分类。到20时候可以完美的把初音的樣子勾勒出來，這個其實沒有甚麼，Decision Tree 只要你想的話，永遠可以做到 Error Rate 是 0 ，永遠可以做到正確率是 100 ，因為你想想看最極端的 case 就是這個 tree 一直長下去。

![1714816084127](https://github.com/joycelai140420/MachineLearning/assets/167413809/2f6211e2-d97a-4003-93f9-d0380d4a6640)

但是因為 Decision Tree 太容易 Overfitting，所以單用一棵 Decision Tree ，你往往不見得可以得到好的結果， 所以要對 Decision Tree 做 Bagging 這個方法就是 Random Forest。Random Forest 比較 typical 的方法是在每一次要產生 Decision Tree 的 branch 的時候 都 random 的決定哪一些 feature 或哪一些問題是不能用。同时，如果是用 Bagging 的方法有一個叫 Out-of-bag 的方法，是可以幫你做 Validation。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b091b83d-8355-4c91-af0c-1f889417328f)

下面是我使用Random Forest进行图形分类并利用Out-of-Bag (OOB) 方法进行验证，在这个示例中，我们将使用著名的Iris花卉数据集，这个数据集包含了三种不同类型的鸢尾花的萼片和花瓣的长度和宽度，来进行分类。可以看到当树的深度到10时候，就完全分类出来。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ef58de43-ffba-4985-8b2d-696bb9175555)
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e157a065-623b-473d-b15f-e298c3ae7f88)
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b353fe8b-e57f-4084-8941-300a0354ff54)


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/c8b8d470-11c7-4e89-9e1b-47d48bd15673)


Boosting 

Boosting 是用在很強的 model ，Boosting 是用在弱的 model 上面，可以将原本錯誤率高過 50% 的 ML classifier， Boosting 這個方法，可以保證最後把這些錯誤率僅略高於 50% 的 classifier 組合起來以後，讓錯誤率達到 0% ，有沒有聽起來非常神奇！！

整個大架構
classifier f1 很弱， 接下來再找一個 classifier f2 ，让他去輔助 f1，但要注意 f2 跟 f1 不可以很像， 他們要是互補。f2 跟 f1 的特性是互補 ，f2 要去彌補 f1 的缺失。f2 要去做 f1 沒有辦法做到的事情，這樣進步量才大。再找一個 f3 跟 f2 是互補的， 接下來再找一個 f4 跟 f3 是互補的，以此类推。這個 process 就繼續下去，再把 classifier 合起來就可以得到很低的 error rate。就算是每個 classifier 都很弱 ，也沒有關係。要注意的是在做 Boosting 的時候， classifier 的訓練是有順序的。要先找出 f1 才找得出 f2 才找得出 f3。

在前面在 Bagging 的時候 ，每一個 classifier 是沒有順序的，在做 Random Forest 要 train 一百棵 Decision Tree， 這一百棵 Decision Tree 可以一起做。但是同样的用Boosting时，要按順序做，它不是平行做的方法 這邊假設我們考慮的一個 task, 是一個 Binary Classification 的 task，
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1e94dad6-f9ec-4141-b962-73bd52be218c)




 Adaboost 

 Boosting 有很多的方法等一下我們要介紹其中最經典的 Adaboost，Adaboost 他的核心思想是将许多弱学习器按顺序迭代训练，每个学习器在训练过程中增加之前学习器分类错误的样本的权重。这样，随着学习器的增加，算法能够更加关注那些难以分类的样本。
![1714826401912](https://github.com/joycelai140420/MachineLearning/assets/167413809/9da68b50-7bf1-4b64-a64e-a8f513bf2999)

更直观的方式，给错误的配分提高，例如第一次答对三题，所以err rate=0.25，接下來要改變 data 的 weight 要把 u 值變一下，他答錯的題目 weight 是 √3 它答對的題目 weight 是 1/√3 有三題 ，他答錯的題目 weight 是 √3 它答對的題目 weight 是 1/√3 有三題，所以这时候f1的err =0.5，这时候f1就变差。接下来根据這組新的 training data 上面再去訓練 f2，那 f2 因為它是看著這組新的 weight、新的配分去做練習的，所以新的 Error Rate 在這組 weight 上，它的 error 會是小於 0.5。所以 f2 可以跟 f1 是互補 。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/83ebf54f-3fd0-4b2d-b33d-f0f94c81e48c)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/61c0018d-5bdc-4a7b-a7da-0f540ff3f5f9)

根据下面的范例，如果有一個比較正確的 classifier ，錯誤率比較低的 classifier ，它得到的 alpha t 的值是大的。如果爛的 classifier 它得到的 alpha t 的值是小的 ， 它當初訓練的時候錯誤率是比較大的 ，它的 weight 就比較小，反之，當初訓練的時候錯誤率比較小它的 weight 就比較大。
![1714829105850](https://github.com/joycelai140420/MachineLearning/assets/167413809/4f456456-e08f-49c8-863a-a23bb6abef2d)

然后使用纯粹的NumPy来实现AdaBoost算法是一种极好的方式来深入理解其背后的原理。下面我们将通过一个简单的二分类问题来演示AdaBoost的实现。我们将使用决策树桩（Decision Stump）作为基学习器，决策树桩是一个一层的决策树，通常用于AdaBoost中。通过运行这段代码，你可以深入理解AdaBoost算法如何逐步聚焦于难以分类的样本，并通过组合多个弱学习器来提升模型整体的分类性能。请参考AdaBoost.py。




