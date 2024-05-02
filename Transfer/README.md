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


    結果

    只 copy 第一個 layer 的時候，performance 稍有進步；但 copy 越多層 layer，performance 就壞掉了,所以，實驗顯示出在不同的 data 上面，前面幾層 layer 是可以共用的，後面幾層 layer 是無法共用的

    最上面那條橙色的線是「Transfer learning + Fine-tuning」，可以發現在做有的 case 上 performance 都有進步

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9466c578-9483-4811-ba58-d7243ad720bf)
發現

紅色的線：跟上圖中的紅線是同一條

綠色的線：假設參數是 random 時，結果壞掉

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f1babd14-8402-48b4-8450-d6238d542042)







Multitask Learning















