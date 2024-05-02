Transfer Learning

迁移学习（Transfer Learning）是一种强大的策略，它允许模型在一个任务上学到的知识被应用到另一个但通常是相关的任务上。这种技术特别有用，因为它可以显著减少对大量标记数据的需求，尤其是在目标任务上的数据稀缺时。

迁移学习通常涉及两个阶段：

1.预训练阶段：在一个有大量标记数据的源任务上训练模型。这个任务通常与目标任务相似或相关，例如，在ImageNet数据集（包含数百万标记图像的大型数据集）上训练用于图像分类的深度学习模型。

2.微调阶段：将预训练的模型（或模型的一部分）应用到目标任务上。通常，模型的高层特征提取部分保持不变，而最后的几层会针对新任务进行重新训练或微调。

应用

迁移学习在许多领域都有广泛的应用，尤其是在那些数据收集成本高或数据稀缺的领域，比如：

  1.计算机视觉：在基础图像识别任务上预训练模型，然后将其迁移到特定的视觉识别任务，如面部识别、医学影像分析。就例如辨别狗猫，训练时给的是真实狗猫图片，但test用的是卡通图片等。

  2.自然语言处理：使用在大规模文本数据集上预训练的语言模型（如BERT、GPT），然后迁移到特定任务，如情感分析、文本摘要、问题回答等。

  3.语音识别：在通用语音数据上预训练语音识别模型，然后针对特定口音或噪声条件进行微调。

  4.无监督学习：在大量未标记的数据上训练模型，以学习有用的特征表示，然后将这些特征应用于其他任务。

优点

效率提升：迁移学习可以利用预训练模型快速提升新任务的性能，尤其是当新任务的数据量有限时。

训练时间缩短：由于模型已经在相关任务上学习了有用的特征，因此训练时间比从头开始训练要短得多。
改善性能：对于数据稀缺的任务，迁移学习通常能达到比完全独立训练更好的性能。

缺点

负迁移风险：如果源任务和目标任务不够相关，迁移学习可能会导致性能下降，这称为负迁移。

调参复杂性：决定哪些部分的模型应该被冻结或重新训练，以及如何调整微调的学习率等参数，可能需要大量的实验和领域知识。

过度依赖预训练模型：过分依赖广泛使用的预训练模型可能会限制模型处理特定问题的能力，尤其是在那些预训练模型未见过的新领域或问题上。

接下来就是台大老师Hung-yi Lee授课内容

Overview

使用時機：利用一些 與 task 不直接相關的 data 來幫助現在要進行的 task

舉例：input domain 相似，但 task 不一樣；task 相似，但 input domain 不一樣

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3004af0c-8837-4ce1-bb68-463fa52b9bc5)

使用原因：有些 task 的 data 是較少的，我們可以試著用其他語言的 data 來 improve 原本的 task

舉例：要做台語的語音辨識，但台語的資料量少，我們可以嘗試用中文、英文等其他語言的 data 來 improve 台語的 task

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1368e9d5-32b6-469e-bcad-89680292712a)

舉例：現實生活中的 transfer learning，研究生的生活可以參考漫畫「爆漫王」

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e20c4ba5-1452-44c4-a55c-ec7463aa0e7f)

名詞介紹

    target data：跟要做的 task 有直接相關的 data，可能是 labelled 或 unlabelled

    sorce data：跟 task 沒有直接相關的 data，也可能是 labelled 或 unlabelled

故可將 transfer learning 分成四個象限討論，后面分别根据这四象限做出相应方法的介绍。

先介紹 sorce data （没有很直接任务关系）及 target data 都 laballed 的情況下，最常見也最簡的就是對 model 做 Fine-tuning

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a09a362a-9bef-4021-956a-f171ff5eddc7)


Model Fine-tuning

概覽

    使用時機：有 labelled 的 大量 source data(x^s, y^s) 及少量 target data(x^t, y^t)

    想法：我們想知道在 target data 很少的情況下，一大堆不相干的 source data 有沒有可能對task 有幫助

    作法：使用 source data 去 train 一個 model，再用 target data 去 fine-tuning 這個 model；亦即將 source data training 出的 model 當作初始值，再用 traget data做 training。

    One-shot learning：如果 target data 的量非常少，少到只有幾個 example，就叫做 One-shot learning

    舉例：語音上，最典型的例子是 speaker adaption，target data 是某一個人的聲音，source data 是一大堆來自不同人的 audio data

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ef7b85d8-f10a-4207-bf43-020df279348a)

Conservative Training：Fine-tuning 時的技巧之一

    作法：在 traget data training 時加一些 constrain (即 regularization)，讓 train完後的新 model 跟原本的舊 model 不要差太多，這樣可以防止 overfitting，防止如果 target data 非常少，一 train 就壞掉的          情況。

    舉例：可以加 constrain 讓新 model 跟舊 model 看到同一筆 data 的時候，output 越接近越好；或是 L2-norm 的差距越小越好

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/246c56d6-f228-41fc-8d6e-afe16156a6de)

Layer Transfer：Fine-tuning 的技巧之二

作法：將 source data train 好的 model 中某幾個 layer 拿出來，直接 copy 到新的 model 中，再用 target data train 剩下 (沒有copy) 的 layer；如果 target data 夠多，要 fine-tuning 整個 model 也是可以的

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/b57aa04b-7979-4ce8-998a-782998415ddc)

技巧：如何選擇哪些 layer 應該被 trasfer，哪些不應該被 transfer？

    語音辨識：通常是 copy 最後幾層，重新 train input 那幾層

    前幾層：從聲音訊號得知說話者的發音方式，而每個人的口腔結構不同，同樣的發音方式，得到的聲音是不一樣的

    後幾層：根據發音方式判斷現在說的是哪一個詞彙，即可得辨識的結果，這個過程跟說話者較無關



    圖片辨識：通常是 copy 前面幾層，重新 train 最後幾層

    前幾層：往往是 detect pattern，如直線、橫線、幾何圖形等，可被 transfer

    後幾層：往往是比較抽象的概念，沒有辦法 transfer

故不同的 task，需要被 transfer 的 layer 往往是不一樣的，需要一些 domain know-how 來幫助判斷
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0f3171c3-5717-4458-8e92-d76443170570)

舉例：image 在 layer transfer 上的實驗，出自 Bengio 在 NIPS, 2014 的 paper

    定義

    橫軸：代表做 transfer learning 時 copy 的 layer 數目

    縱軸：為 Top -1accuracy，越高代表表現越好

    Data：將 ImageNet 中 120 萬張 image，其中 500 個 class 歸為 source data，另外 500 個 clasee 歸為 target data

    Baseline：圖中空白圓點，完全沒做 transfer learning


    結果只 copy 第一個 layer 的時候，performance 稍有進步；但 copy 越多層 layer，performance 就壞掉了,所以，實驗顯示出在不同的 data 上面，前面幾層 layer 是可以共用的，後面幾層 layer 是無法共用的

    最上面那條橙色的線是「Transfer learning + Fine-tuning」，可以發現在做有的 case 上 performance 都有進步


![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9466c578-9483-4811-ba58-d7243ad720bf)
可以参考范例Conservative Training.py,使用了CIFAR-10图像数据集，这是一个常见的用于图像识别任务的数据集。我们将加载一个预训练的模型（这里以VGG16为例），然后对其进行Fine-tuning以分类CIFAR-10中的图像。透过以下是结果，还是有部分有优化空间可以参考以下的几个方向，在这里因为硬件关系，就不做优化范例

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/814ef350-696b-4781-a30b-cb76bef36171)

我们优化模型的方向有：
1. 解冻一些层
    虽然一开始我们冻结了预训练模型的所有层，以防止丢失在预训练阶段学到的特征，但你可以尝试解冻一些顶层（靠近输出层的层）。这些层通常更加特定于原任务，通过对它们进行微调，可以使模型更好地适应新任务。
    操作方法：选择从倒数第二、第三层开始解冻，观察模型性能的变化。
2. 增加数据增强
   在迁移学习中，特别是当目标任务的数据量不足以支持复杂模型时，数据增强可以显著提高模型的泛化能力。
   操作方法：可以使用旋转、平移、缩放和水平翻转等技术来增强图像数据。在TensorFlow和Keras中，这可以通过ImageDataGenerator类轻松实现。
3. 调整学习率和优化器
   调整学习率和改变优化器策略可以帮助模型更有效地收敛。
   操作方法：开始时使用较低的学习率，并逐步调整。尝试使用不同的优化器，如Adam、RMSprop等，以找到最适合当前数据和模型结构的配置。
4. 使用正则化技术
   应用正则化方法，如L2正则化或Dropout，可以减少过拟合，尤其是在数据较少的情况下。
   操作方法：在全连接层中加入Dropout层或者使用带权重衰减的Dense层。
5. 增加/减少模型复杂性
   根据目标任务的需求调整模型复杂性。如果模型过于简单，可能无法捕捉到所有有用的特征；如果模型过于复杂，可能会过拟合。
   操作方法：尝试添加或移除一些层，或者调整每层的神经元数量。
6. 增加训练周期
   有时简单地增加训练的轮数（epochs）可以提高模型的性能。但要注意监控验证损失，以避免过拟合。
   操作方法：逐步增加epochs数目，同时使用早停（Early Stopping）来监控验证集上的性能。
7.使用更多或更相关的预训练模型 
   如果可用，尝试使用与目标任务更紧密相关的预训练模型，或者尝试使用最新的模型，这些模型可能具有更好的特征提取能力。
   操作方法：探索不同的预训练模型，比如在计算机视觉任务中，除了VGG和MobileNet之外，还可以考虑ResNet、Inception等。

   
發現

紅色的線：跟上圖中的紅線是同一條

綠色的線：假設參數是 random 時，結果壞掉

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f1babd14-8402-48b4-8450-d6238d542042)


Multitask Learning

在Fine-tuning我们只在意做得好不好，不在意sorce data是否烂掉，但在Multitask Learning中都会考虑进行。

概覽

    使用時機：同時關心 target domain 及 source domain 的表現，希望兩者表現都好

    訓練模型：Deep learning based 的方法，特別適合用來做 multitask learning

    舉例：
    (左圖)：兩個不同的 task，相同的 input feature，只是影像辨識的 class 不同 我們就可以 learn 一個 NN，input 相同的 feature，但是中間岔開分別 output task A 及 B 的結果。 這麼做的前提是這兩個 task 有共通性；優點是前面幾個 layer 使用較多的 data train，表現可能較好
    (右圖)：兩種不同的 task，不同的 input feature，使用不同的 NN 我們可以把它 transform 到同一個 domain 上，這樣中間幾個 layer 也可以做 multitask learning，最後，再 apply 到不同的 NN，分別 output 各自的結果

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d2588844-2ce8-4d01-b7a3-571e570c7932)

應用：Multitask learning 一個很成功的例子是「多語言的語音辨識」

    Input 是各種不同語言的 data，在 train model 時，前面幾層的 layer 會共用參數，後面分岔；因為不同的語言都是人類說的，所以前面幾層可以 share 同樣的資訊。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9451c0f0-6063-4410-affc-c387d8418d2f)

推廣：「翻譯」也可以做同樣的事

    在中翻英、中翻日時，因為都要先 process 中文 data 的部分，因此一部分的 network 就可以共用。這種語言 transfer 的範圍有多大呢？目前的發現是幾乎所有的語言都可以 transfer
    藍線是只有中文 data training 的結果，橘線是從歐洲語言 transfer 到中文上面，可以發現 單獨 train 中文 100 小時的 performance 跟 train with 歐洲語言 50 小時的效果是一樣的， 亦即在這個例子中，我們只需要 1/2 以下的 data，就可以跟原來有兩倍的 data 的效果一樣好。所以你也可以中翻译成台语、沪语、高山族语。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/458b1f2d-1f05-492f-a742-03a21d11ea13)

Progressive Neural Network：
如果有两个task不像的化，Transfer就是有可能negative的，总是思考两个task不像，做Transfer进行try error很浪费时间，所以提出Progressive Neural Network。

    想法：先 train 一個 task 1 的 NN，train 好後，task 2 的每一層 hidden layer 就去接 task 1 某一層 hidden layer 的參數，而 task 2也可以把這些參數直接設成 0，亦即最糟的情況跟 task 2自己 train 的 performance 是一樣的。而 task 3 再從 task 1 及 task 2 的 hidden layer 得到 information

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/228dff16-954d-4ae9-9c11-7e249e432411)

多任务学习（Multitask Learning）是一种机器学习方法，它通过在模型中共享表示层来同时解决多个相关任务，从而可以提高各任务的学习效率和预测性能。在这种设置中，不同任务的共通特征被学习一次，但每个任务有其专有的输出层。我们可以参考Multitask learning.py范例。演示如何设置一个简单的多任务学习模型，该模型将同时对图像进行分类和回归分析。我们将使用修改后的MNIST数据集，其中分类任务是识别手写数字，而回归任务则是预测图像的平均像素值。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/57b4ce89-6c7a-4090-91b1-e237f7c95772)





Domain-adverserial Training
概覽

    使用時機：資料是 labelled source data 及 unlabelled target data 時

    source data: (x^s, y^s), target data: (x^t)

    舉例：以下圖為例，source data 是 MNIST 上有 labeled 的 image，target data 是 MNIST-M 上沒有 labeled 的 image 這種情況下，我們通常將 source data 視為 train data，target data 視為 testing data 但遇到一個問題是：training data 跟 testing data 非常的 mismatch

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ed276d98-3326-4426-acf1-2c1ad1c54448)

實作

想法：

    如果直接 learn 一個 model，如下圖，會發現前面幾層抽 feature 的結果是爛的。 藍色很明顯地分成十群，紅色這群卻是一坨。
    在 feature extraction 時，source domain 跟 target domain 不在同一位置上，所以，我們希望 能把 domain 的特性去除掉 。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/6397c214-c4e3-4647-89f2-6a9025f97f3d)

實現：

    在 feature extractor 後面接一個 domain classifier，將 output 丟到 domain classifier，如此一來，就能將 domain 的特性消掉，將不同 domain 的 image 混在一起
    而有一個 generator 的 output 跟有一個 discriminator 這樣的架構，非常像 GAN，但在 domain-adversarial training 中，要產生一張 image 騙過 classifier 太簡單了，所以 feature extractor 的 output 不只要騙過 domain classifier 還要同時讓 label predictor 做好

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/beda1a16-4a18-461b-9f8e-3d089f043006)

domain-adversarial training 三個 part 的目標是各自不同的

    Feature extractor : 增進 label predictor 正確率的同時，最小化 domain classifier 的正確率
    Label predictor : 最大化 classification 的正確率
    Domain classifier : 正確的預測一個 image 屬於哪一個 domain

所以，feature extractor 跟 domain classifier 想做的事情是相反的

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/aa32ff27-0d96-4471-b311-43431d9d1300)

Feature extractor 只要在最後加上一個 gradient reversal 的 layer； 這樣 domain classifier 做 back propagation，計算 backward path 時，feature extractor 會將 domain classifier 傳進來的 output 故意乘上負號，做跟 domain classifier 要求相反的事

這樣，domain classifier 會因看不到真正的 image，而 fail 掉 但是，domain classifier 一定會 struggle 完才 fail，這樣才能把 domain 的特性去掉

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/92f81250-e973-4d33-8ef7-23b68a3b625a)

例子

引用 ICML,2015 跟 JMLR,2016 的 paper 中一些實驗的結果

概覽

    資料：包括 MNIST 到 MNIST-M、一個數字的 corpus 到另一個數字的 corpus、數字的 corpus 到 MNIST 及兩種不同道路號誌 data 互相 transfer 等
    縱軸：每筆資料用四種不同方法得到的實驗結果
    
比較

    1.Source only 是直接在 source domain 上 train 一個 model，再到 testing domain 上 test
    2.Proposed 的即 domain-adversarial training
    3.Train on target 則是直接拿 target domain 的 data 去做 training，得到的 performance 即 upper bound
    
發現

    source only 跟 train on targert 間有很大的 gap，而用 domain-adversarial training 在不同的 case 上，都可以得到很好的 improvement

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7df74049-aeff-47cf-8965-f8f711773f68)


Zero-shot Learning
概覽

    使用時機：資料是 labeled source data 及 unlabeled target data，且 source data 跟 target data 要做的的 task 是不一樣的

舉例：

    如下圖，source data 要分辨貓跟狗，target data 的 image 則是草泥馬

    最常見的則是語音上的應用

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/adbb6219-2ec5-4b43-b147-0334869d04c0)

實作

Representing：

    將每一個 class 用他的 attribute 表示； 亦即，有一個 database 存每一個 obeject 跟所有可能的特性（如下圖右下表），ex: 狗：毛茸茸、四隻腳、有尾巴；魚：不毛茸茸、沒有四隻腳、有尾巴......等

    每個 class 都要有獨一無二的 attribute，亦即 attribute 要定的夠豐富才行 一旦有兩個 class 有一模一樣的 attribute，這個方法就會 fail 掉

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/20aa565b-2f4f-4872-87b3-93423d9a736d)

Training & Testing

   在 training 時，要做的是辨識每一張 image 具備什麼樣的 attribute，而不是直接辨識這張 image 屬於哪一個 class；所以即使 testing 時出現一張不存在的動物，我們只要辨識出它具有哪些 attribute，查 database 找最接近的動物，就可以了

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/d2165200-00ef-4002-91b7-3c3d6fad66b5)

Attribute embedding

   當 attribute 的 dimension 很龐大時，我們可以做 attribute 的 embedding
   將 training data 上的每一張 image 跟每一個 attribute 都 transform 成 embedding space 上的一個點 f(x^1), g(y^1)
   f,g 都可以是 neural network，training 時希望 f(x^n) 跟 g(x^n) 越接近越好
   出現一張沒看過的草泥馬 x^3 時，就投影到 f(x^3)，再找最近的 g(y^3)，y^3 就是他的 attribute，再看對應到哪一個動物，就結束了

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/be0fb62a-0760-4751-9410-6c560c157c93)

Convex combination of semantic embedding

條件：有一組 word vector + 一個語音辨識系統

作法：

   將圖片丟到 NN 中，得到有0.5的機率是獅子、0.5的機率是老虎
   再找獅子跟老虎的 word vector 用上面的比例(1:1)混合，得到新的 vector
   再找哪一個 word 跟混合出的新 vector 最接近， 最後得到獅虎，這張圖片就是獅虎

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/829e088b-8926-43ab-a9a2-61d578eafd18)





Self-taught Learning
概覽

使用時機：資料是 unlabeled source data 及 labeled target data

作法：

   用夠多的 unlabeled source data 去 learn 一個 feature extractor
   再用這個 feature extractor 在 target data 上抽 feature

![Uploading image.png…]()

