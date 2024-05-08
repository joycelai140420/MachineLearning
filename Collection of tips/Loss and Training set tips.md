在学习的过程中，总是会零零散散一些非常小的关键小知识，在这里我会统计整理攸关一些loss 的小技巧与知识点。至于技巧与知识点的推理演算过程，会比较少，如果能找到文献我会放，这里就不对这个过程做太多深入的探讨。


参考台大老师Hung-yi Lee课程

1.L(h^train,Dall) <= L(h^train,Dtrain)+&/2 <= L(h^all,Dall)+&/2+&/2 = L(h^all,Dall)+&/2
得：
L(h^train,Dall)-L(h^all,Dall) <= &

当你做出训练好模型时，大概率可以推出使用所有真实数据的loss function。让理想跟现实很接近。

说明：

    Dtrain是Training set data

    h是分类的阈值（filter 的阈值）

    h^train就是在Training set data 训练出来的分类阈值（filter 的阈值）

    Dall就是real all data

    h^all就是在real all data data 训练出来的分类阈值（filter 的阈值）

    L就是loss function

    &就是你期望训练与真实结果的差距

![1715043316215](https://github.com/joycelai140420/MachineLearning/assets/167413809/1f644511-8e6c-4c99-90f5-1585a4aa4ea5)

这个假设适用于deep learning，跟模型不相关，也不假设任何data 分布，也可以适用于任何的loss function。

2.定义e=&/2，Dtrain是坏的，找随机的一个h，则，|L(h,Dtrain)-L(h,Dall) | > e。

![1715045814893](https://github.com/joycelai140420/MachineLearning/assets/167413809/24066503-05eb-4686-858c-a7f5bd216b66)


在这边我们可能对h关联很模糊，可以参考台大林轩田老师举的核发信用卡给申请者的范例。h是假设，假设1:到底這個人的年收入有沒有超過80萬台幣， 有超過80萬台幣我們給他信用卡，沒有超過80萬台幣我們不要給他信用卡。假设2:如果這個人負債超過10萬就給信用卡。假设3:如果在工作不滿兩年的話。这些全部的假设就是h，那么H就是所有假设的集合。

![1715146508814](https://github.com/joycelai140420/MachineLearning/assets/167413809/310327b2-8ebb-4da9-ae53-d58d6375fd54)


3.训练集会差的几率 = h1 union h2 union h3， 但无法考虑overlap ，所以就直接重覆计算overlap就是会训练集会差的几率的upper bound 上限。但upper bound有可能超过 1，哈哈。


![1715046170693](https://github.com/joycelai140420/MachineLearning/assets/167413809/d89bd143-202c-442f-8718-45f63d1af255)

![1715049229738](https://github.com/joycelai140420/MachineLearning/assets/167413809/c927a2e7-f8d3-47d5-8ebf-2f2653b97138)

因  
    e=&/2，
    N为所有Dtrain的examples的number数 （分很多train 资料的数目,所以在之前课程又说到N-fold cross Validation，在MachineLearning/Regression
/README.md
 ）

又
    h让训练集会差的几率 小于等于 2exp(-2Ne^2)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7a005d9d-1077-4730-bc20-989f42b0daef)

又，h让训练集会差的几率 小于等于 你可以选择的function 数目绝对值 * 2 exp(-2Ne^2)

![1715049678222](https://github.com/joycelai140420/MachineLearning/assets/167413809/95807d4b-cfe0-425d-86de-40deb976880f)

怎么让训练的几率变低呢？所以只要让N越大，sample 坏的记录就变低。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2baaee90-f57d-4920-82c6-d8d81a87e90a)


如果你将|H|（可以选择的function 数目）变少，则，sample 到坏的记录就变低。这里的H可以用VC-dimension(参考台大林軒田 老师内容)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1d6a6974-47f1-44e2-86bd-839d1aeb59d9)

![1715051061660](https://github.com/joycelai140420/MachineLearning/assets/167413809/c17d0f65-353e-495b-ab5d-6abdf538a655)

根據上面的P（Dtrain is bad) 小于等于 |H| * 2 exp(-2Ne^2) ，如果我們H用 validation set 也是有可能讓訓練變差，因為可能選擇的validation set也不好，造成overfitting。

那validation set選到差取決於:
        1.data 大小，
        2.validation set對應的H複雜程度

