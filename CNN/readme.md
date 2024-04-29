CNN (卷积神经网络) 
CNN 通过使用卷积层来有效捕捉空间和时间上的局部相关性(局部/捕捉就是指不是全部看整张图，可能只看关键可立即辨识出的关键小图片)，适用于图像和视频识别、推荐系统、图像分类、医学图像分析等领域。

CNN 的核心组成部分

卷积层（Convolution）：通过滤波器（或称为卷积核）提取输入数据的特征。

池化层（Max Pooling）：也称为下采样层，用于减少数据的空间尺寸，从而减少参数数量和计算复杂度，防止过拟合。

优点：

参数共享：卷积层中整个输入数据上共享，大大减少了模型的参数数量。

局部连接：每个神经元只与输入数据的局部区域相连接，使得网络能够捕捉局部特征，并且具有更好的空间或时间的泛化能力。

适合图像处理：能够直接处理像素数据，不需要额外的特征工程。

缺点：

相对复杂和计算量大：尤其是当网络层数较深时。

可以参考Keras_helloworld_CNN.py

以下内容来自于台大Hung-yi Lee 教学内容

Structure of CNN
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/20036a48-7f1d-44b3-8d17-7f0418046c6c)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/9cef23c7-15a0-4c0a-8175-8b15ba1b00cc)

Part 1 - Convolution
黑白圖像（0 代表白色，1 代表黑色）

Filter：Convolution layer 裡有許多組 filter，一個 filter 其實就是一個 matrix

Filter 中的參數是根據 training data 學得的，3*3 的 size 代表偵測一個 3*3 的 pattern

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7391d7b5-719b-41de-968a-39b3c8c03a3c)

內積：filter 從左上角開始放，與 image 中對應的值做內積；如下圖，最左上角與 filter 做內積後得到 3
Stride：挪動 filter 的位置，移動多少稱作 stride；如下圖，紅色框框一次偏移 2 格，stride = 2 

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ac349693-e83c-45ab-bd34-d0209329f6d3)

偵測：觀察 filter，他要偵測的 pattern 為圖中是否有出現「左上到右下斜線的值為 (1, 1, 1)」

亦即，內積完後，出現最大值的地方，就代表這個 filter 要偵測的 pattern

Property 2：做此步驟時，有考慮 Property 2（同樣的 pattern 可用相同的 filter 偵測出來）

如下圖，左上角及左下角出現了同樣的 pattern，我們用同一個 Filter1 即可偵測出來

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0916115b-edef-4f32-8547-9b1c6cb2c971)

彩色圖像（RGB）

彩色圖像就是好幾個 matrix 疊在一起，是一個立方體；故我們使用的 filter 也是「立方體」。 計算時，並非每個顏色（channel）分開算，而是每個 filter 同時考慮了不同顏色代表的 channel

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4279c9ea-fca0-4c45-a67a-a2850d1fc52c)

Convolution 與 Fullu Connected 的關係

Convolution 其實就是一個 neural network
Convolution 就是 Fully Connected 的 layer 將一些 weight 拿掉
Feature Map 這個 output 就是 Fully Connected neuron 的 output（綠色框框互相對應）

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/712f88f9-2da8-4154-9e60-bc98cef3de9f)

Convolution 與 Fullu Connected 相對應

Convolution 的 filter 放在左上角，考慮的 pixel 是 [1, 2, 3, 7, 8, 9, 13, 14, 15]
將左邊 6*6 的 Image 拉成直的，filter 做內積後得到 3，即右圖某個 neuron 的 output 為 3
左圖 filter 代表的意思為，右圖 neuron 的 weight 只連接到 [1, 2, 3, 7, 8, 9, 13, 14, 15] 這些 pixel 而已

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/7f3fd091-0830-4e25-adba-151ddb1909f0)
Stride = 1

filter 移動到下一格，得 -1，亦即另外一個 neuron 的 output 為 -1
這個 neuron 就只連接到 [2, 3, 4, 8, 9, 10, 14, 15, 16] 這些 pixel
在 Fully Connected layer 中，每個 neuron 本來都有自己獨立的 weight；當我們做 convolution 時

減少每個 neuron 前面連接的 weight
強迫某些 neuron 要共用某一組 weight（如下圖中的 pixel 2, pixel 3 等）即 Shared weights

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/da06520c-72fa-4d37-b636-233b90f2d1b2)

Part 2 - Max Pooling

根據 Filter 1, Filter 2，我們可以得到兩個 4*4 的 matrix

將他們 4 個一組（如下圖），接著每組中選出一個數（可選平均、最大值......自行定義）
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/a8c84889-8a03-46c7-bab8-2b0d25fe7732)

這裡，我們選最大值，故做完一次 Convolution + Max Pooling

我們將原來 6*6 的圖像化簡成一個 2*2 的圖像

Filter1=[ 3 0
         3 1 ]
              
Filter1=[ -1 1
          0  3 ] 


image 深度(維度)：由「filter 個數」決定

例：這裡只有 Filter 1 及Filter 2，故右下圖中的維度就是 2

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4ca73371-3718-4a66-a295-e43787e7b287)

反覆「Convolution + Max Pooling」的動作 我們可以重複「Convolution + Max Pooling」的動作，讓 image 越來越小

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2e15160a-0b89-425a-ae28-b1f2df54f14d)


Part 3 - Flatten
將 Feature Map 拉直，丟到一個 Fully Connected Feedforward Network 中，就結束了

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ce0356f1-233b-4d0b-828e-67e5c823cdde)


CNN tips:

训练卷积神经网络（CNN）是一个复杂的过程，其中包含了许多可以调整的部分以改进模型的性能。下面我将提供一些提高训练过程中准确率的方法，以及在使用测试集和一般情况下如何提升模型的准确率。我们透过DNN里面有讲解到几个可改进的方向，在CNN也可以同样参考那样的思路。

1. 训练过程中的优化策略
   数据预处理
         标准化：确保输入数据标准化（0-1范围）或归一化（均值为0，方差为1）可以帮助模型更快地收敛。
         数据增强：通过随机变换（如旋转、缩放、翻转、裁剪等）可以增加训练数据的多样性，这有助于模型学习到更加鲁棒的特征，减少过拟合。在 Keras 中可以使用                   ImageDataGenerator 来实现。
   网络架构调整
         增加网络深度：适当增加网络的层数可以帮助网络捕捉更复杂的特征。
         使用预训练模型：在可能的情况下，使用预训练的模型作为起点，通过迁移学习进行微调，这通常可以显著提高性能。
   超参数优化
         优化器选择：试验不同的优化器（如 Adam, RMSprop 等），每种优化器都有不同的特点和表现。
         学习率调整：学习率是最重要的超参数之一。可以使用学习率衰减或通过实验找到最佳的学习率。
         正则化技术：应用 L1、L2 正则化或 Dropout 可以减轻模型在训练数据上的过拟合。
   批次大小和迭代次数
         批次大小：较小的批次可以提供更稳定的误差梯度估计，但同时增大了计算开销。
         迭代次数：确保足够的迭代次数让模型完全学习，使用早停（early stopping）策略防止过拟合。
2. 提升测试集上的准确率
   模型评估策略
         交叉验证：使用 K 折交叉验证可以更全面地评估模型的泛化能力。
         集成学习：通过集成多个模型的预测来提高测试集上的准确率，例如通过模型融合或集成不同的 CNN 架构。

