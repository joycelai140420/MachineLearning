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

