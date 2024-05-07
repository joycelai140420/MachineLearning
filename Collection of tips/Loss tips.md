在学习的过程中，总是会零零散散一些非常小的关键小知识，在这里我会统计整理攸关一些loss 的小技巧与知识点。至于技巧与知识点的推理演算过程，会比较少，如果能找到文献我会放，这里就不对这个过程做太多深入的探讨。


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

参考台大老师Hung-yi Lee课程
![1715043316215](https://github.com/joycelai140420/MachineLearning/assets/167413809/1f644511-8e6c-4c99-90f5-1585a4aa4ea5)

这个假设适用于deep learning，跟模型不相关，也不假设任何data 分布，也可以适用于任何的loss function。

2.定义e=&/2，Dtrain是坏的，找随机的一个h，则，|L(h,Dtrain)-L(h,Dall) | > e。
参考台大老师Hung-yi Lee课程

![1715045814893](https://github.com/joycelai140420/MachineLearning/assets/167413809/24066503-05eb-4686-858c-a7f5bd216b66)

3.训练集会差的几率 = h1 union h2 union h3， 但无法考虑overlap ，所以就直接重覆计算overlap就是会训练集会差的几率的upper bound 上限。但upper bound有可能超过 1，哈哈。


![1715046170693](https://github.com/joycelai140420/MachineLearning/assets/167413809/d89bd143-202c-442f-8718-45f63d1af255)

![1715049229738](https://github.com/joycelai140420/MachineLearning/assets/167413809/c927a2e7-f8d3-47d5-8ebf-2f2653b97138)

因  
    e=&/2，
    N为所有Dtrain的examples的number数 （分很多train 资料的数目,所以在之前课程又说到1 of N）

又
    h让训练集会差的几率 小于等于 2exp(-2Ne^2)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7a005d9d-1077-4730-bc20-989f42b0daef)

又，h让训练集会差的几率 小于等于 你可以选择的function 数目绝对值 * 2 exp(-2Ne^2)

怎么让训练的几率变低呢？所以只要让N越大，sample 坏的记录就变低。
![1715049678222](https://github.com/joycelai140420/MachineLearning/assets/167413809/95807d4b-cfe0-425d-86de-40deb976880f)

