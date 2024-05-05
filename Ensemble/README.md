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


老師也用一個簡單的範例，明白Boosting操作。
剛剛的演算法如果沒有聽懂就看這個例子 就知道它的意思了 假設大 T = 3，現在 weak 的 classifier 很 weak， 它不是 Decision Tree 也不是 Neural Network，還是很弱的 decision stump ，它做的事情就是 假設 feature 都分佈在二維平面上 ，在二維平面上選一個 dimension 切一刀，從實際意義來看，decision stump 根據一個屬性的一個判斷就決定了最終的分類結果，比如根據水果是否是圓形判斷水果是否為蘋果。其中一邊當作 class 1 ，另外一邊當作 class 2 結束，這個就叫做 decision stump。

一開始每一筆 training data 的 weight 都是一模一樣的都是 1.0。用 decision stump 找一個 function， 這個 function 是 f1，它的 bounder 就切在這個地方， 以左就說是 positive。一邊 class 1 是 positive 的 ，往右就是粉紅色就是 negative 的。然後根據左圖可以看到還是有三個data是分類錯誤。data 總共有10筆，所以三筆 data 分類錯，其 Error Rate 是 0.3，d1 算出來就是 1.53， alpha 算出來就是 0.42，是根據上一頁的公式得到。

現在已經算出 epsilon 1, d1, alpha 1 ，接下來就是改變每一筆 training data 的 weight，分類正確的 weight 就要變小 ，分類錯誤的 weight 就要變大。分類錯誤的要乘 1.53 ，分類對的就要除 1.53 ，讓這三筆分類錯的 weight 變大。分類對的 weight 就變小 ，就有了一組新的 weight 。

![1714870195705](https://github.com/joycelai140420/MachineLearning/assets/167413809/9d63752a-0128-4ef8-bae7-f693ae1be34a)

然後再去找一次另外一個 decision stump，有一組新的 weight ，找出來的 decision stump 就不一樣了！ 在新的 decision stump 切一刀切在這個地方，往左是 positive 往右是 negative， 往左是藍色往右是紅色，會發現有三筆 data 紅色的分類是錯的， 會根據每一筆 data 的 weight 進行計算， 就會發現第二個 classifier 的 Error Rate 是 0.21，它的 d2 = 1.94, alpha 2 = 0.66 ，接下來這三筆 data 分類錯所以給他 weight 比較大，這三筆 data 要把它乘上 1.94， 剩下的 data 把他除掉 1.94。就找到了第二個 classifier ，每一個 classifier 的 weight 就是它 alpha 的值。

![1714870400216](https://github.com/joycelai140420/MachineLearning/assets/167413809/adf3011a-235f-4cc0-a2f3-4f31a66141c6)

把 alpha 的值寫再 classifier 的旁邊， 接下來找第三個 classifier，第三個 classifier 上面是藍色下面是紅色，它這麼講會導致有三筆 data 錯誤，計算一下它的 Error Rate = 0.13 ，然後可以計算它的 d3 和 alpha 。如果有更多 iteration 的話，就會去重新 weight data。但現在只跑三個 iteration 跑完就結束了 得到三個 classifier ，還有他們的 weight 就結束了。

![1714870536277](https://github.com/joycelai140420/MachineLearning/assets/167413809/21dcc1f3-9f20-4c9e-875e-561ed505443f)

最後怎麼把這三個 classifier 組合起來， 把每個 classifier 都乘上對應的 weight，通通加起來再取它的正負號 ，那麼這個加起來的結果到底是怎麼做呢？有三個 decision stump ，這三個 decision stump 把整個二維的平面切成六塊，左上角三個 classifier 都覺得是藍的 ，所以就藍色。中間這一塊他們兩個覺得是藍的，第一個覺得是紅的。但是他們兩個合起來的 weight 比較大 ，所以上面這組就是藍的。右上角第一個覺得是紅的，第二個覺得是紅的，第三個覺得是藍的， 這兩個紅的 weight 合起來比藍的 weight 大 ，所以又是紅的。左下角是第一個藍的，第二個藍的，第三個紅的，兩個藍的合起來比紅的大，所以是藍的 ，下面這個紅的藍的紅的，兩個紅的加起來比藍的大，所以是紅的， 右下角三個 decision stump 都說是紅的，所以是紅的。這樣分，三個 decision stump 沒有一個是 0% 的 Error Rate。他們都有犯一些錯 ，但把這三個 decision stump 組合起來的時候，它告訴我們這三個區塊屬於藍色、這三個區塊屬於紅色， 而它的正確率是 100%。

![1714870813983](https://github.com/joycelai140420/MachineLearning/assets/167413809/a61afea1-94a4-4589-8cb0-7e9170c8c339)



神秘現象

![1714895586620](https://github.com/joycelai140420/MachineLearning/assets/167413809/e5f14a48-b3ea-4e46-8c88-8c81f0da632d)

縱軸是 Error Rate，橫軸是 training 的 iteration，比較低的這條線是在 training data 上的 Error Rate，比較高的這條線是在 testing data 上的 Error Rate，神奇的是 training data 的 Error Rate，其實很快就變成 0， 大概在 5 個 iteration 之後 ，找五個 weak 的 classifier combine 在一起以後，Error Rate 其實就已經是 0 。但神奇是training data 上10次基本都是0，但 testing data 上的 Error Rate還在持續下降。

為甚麼 我們來看一下下面這個式子，最後找到的 classifier 叫 H(x) ，它是一大堆 weak classifier combine 後的結果。把 weak classifier combine 後的 output 叫作 g(x) ，把 g(x) 乘上 y hat。定義為 margin 我們希望 g(x) 跟 y hat 是同號，如果是同號分類才正確 ，不只希望它同號，希望它相乘以後越大越好。不只是希望這個 g(x)， 如果 x 是 positive，如果 y hat 是正的， 不只希望 g(x) 就是稍微大於 0 0.000001。希望它比 0 大的越多越好。如果 y hat 是正的，g(x) 是 0.00001，那一點的 error 就會讓分類錯誤 ，只要一點 training data、testing data mismatch 就會讓分類錯誤。但如果 y hat 是正的，而 g(x) 是一個非常大的正值。

那 error 的影響就會比較小 如果從現象上來看一下 Adaboost margin 變化的話，會發現如果只有五個 weak classifier 合在一起 margin 的分佈是這個樣子。如果有一百個甚至一千個 weak classifier 結合在一起的時候， 它的分佈就是黑色的實線。

五個 weak classifier 的時候就已經不會再下降， 因為所有的 training data 它的 g(x) * y hat 都是大於 0，會發現 margin 的分布都是在右邊 ，也就是 y hat 跟所有的 g(x) 同號。但在加上 weak classifier 以後，可以增加 margin，增加 margin 的好處是讓你的方法比較 robust 可以在 testing set 上得到比較好的 performance。

![1714896143488](https://github.com/joycelai140420/MachineLearning/assets/167413809/235ab9e8-d9b9-4aa4-bff1-09cefc8e68dc)







