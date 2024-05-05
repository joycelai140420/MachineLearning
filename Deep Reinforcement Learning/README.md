Deep Reinforcement Learning（深度强化学习）原理

深度强化学习结合了深度学习和强化学习的原理，用于解决在高维状态空间中进行决策的问题。它使用深度神经网络来近似强化学习中的策略函数或价值函数，这样可以有效地处理像图像和视频这样的复杂输入数据。

基本原理包括：

    1.Agent（智能体）：在环境中执行操作，收集状态信息和奖励。
    
    2.Environment（环境）：给智能体提供状态信息和奖励。

    3.Policy（策略）：智能体根据策略来决定在给定状态下应采取的行动。策略可能是确定性的，也可能是随机性的。

    4.Reward（奖励）：对智能体采取特定行动的即时反馈。

    5.Value Function（价值函数）：预测从某状态开始，遵循特定策略可获得的累积奖励。

    6.Q-learning：一种强化学习算法，用于学习每个动作的价值（Q值）。

    7.Deep Q-Networks (DQN)：结合深度学习与Q-learning的方法，使用深度神经网络来近似Q值函数。

应用场景

    1.深度强化学习已被应用于多种场合，包括但不限于：

    2.游戏：从Atari游戏到复杂的多玩家环境如StarCraft II，深度强化学习已被证实能在这些环境中达到超越人类的表现。

    3.机器人技术：用于导航、自动驾驶汽车、机器人手臂控制等。

    4.自然语言处理：在对话系统、机器翻译等领域的应用。

    5.推荐系统：优化用户的内容推荐，增加用户满意度和平台收益。

    6.自动化交易：在金融市场上，用于开发交易策略。

缺点

尽管深度强化学习非常强大，但它也有一些明显的缺点：

    1.样本效率低：通常需要大量的数据（经验）来训练可靠的模型。

    2.高方差：小的变更（如初始化）可能导致学习过程的巨大差异。

    3.稳定性和收敛性问题：由于目标和优化策略的不断变化，训练过程可能不稳定。

    4.计算资源需求高：通常需要强大的计算能力，如GPU或TPU，来进行训练。

    5.对奖励设计高度敏感：如果奖励函数设计不当，可能会导致学习到不良行为或意外行为。

优点

显著的优点，使其在许多复杂任务和应用中表现出色:

    1. 深度强化学习通过使用深度神经网络，能够有效处理高维和连续的状态空间。

    2.与需要手工设计特征的传统机器学习方法不同，深度学习框架使得深度强化学习能够从原始输入数据自动提取和学习复杂的特征。这降低了对专家知识的依赖，并增加了模型的通用性和可扩展性。

    3.从感知输入（如像素）到控制输出（如动作选择），无需明确编程中间的决策规则。这简化了从感知到动作的映射过程，并可以在复杂环境中直接学习策略。

    4.通过训练，深度强化学习模型能够在一个任务上学习到的策略泛化到其他相似的任务上，尤其是在使用相似的输入数据结构时。这种泛化能力对于开发多功能的智能系统至关重要。

    5.深度强化学习系统能够通过与环境的持续互动来自我改进其性能，无需人工干预。

    6.由于其能够从复杂环境中学习策略，深度强化学习尤其适合解决传统算法难以处理的非结构化问题。这包括自动驾驶、机器人导航等领域，其中环境可能高度动态且难以预测。

    7.在某些特定任务和游戏中，如围棋、多款Atari视频游戏和复杂的策略游戏（如Dota 2和StarCraft II），经过适当训练的深度强化学习模型已显示出超越顶尖人类玩家的性能。


接下来接跟着台大老师Hung-yi Lee课刚走，其中在添加非课程内容的代码辅助说明


Deep Reinforcement Learning

模型基本架構：

    一個 Agent，一個 Environment

    Environment => Agent

        Observation，可以偵測世界種種的變化

        Observation 又叫做 State，是 環境的狀態，也就是 Machine 所看到的東西。

    Agent => Environment

        Machine 會做一些事情，做的事情就叫做 Action，Action 會影響環境。

    Environment => Agent

        造成影響後會得到 Reward，這 Reward 會告訴它，它的影響是好的，還是不好的。
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4f63b995-8543-4150-b3bf-20f82bc44dcf)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/f2817ac1-baed-4329-965c-0d54a61d6f03)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/59055bce-c655-42ee-9a5c-da8a4a8258cc)

舉例：AlphaGo (下圍棋)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e8b18fd4-02ca-4fd4-840f-21c946e60bca)

架構

    Observation：棋盤，可以用一個 19*19 的 matrix 來描述它

    Take action：落子的位置

    Environment：你的對手，你的落子在不同的位置，會影響你的對手的反應

困難的原因

    多數的時候，得到的 reward 都是零。在你贏了，或是輸了的時候，才會得到 reward。
  
    機器怎麼在只有少數的 action 會得到 reward 的情況下，知道哪些是正確的 action

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/39d7accd-5003-4851-9369-d454b295e23d)
machine 要怎麼學習下圍棋?

    不斷地找某一個對手一直下一直下，有時候輸、有時候贏

    調整它看到的 observation 和 action 之間的關係，讓它得到的 rewards 最大化。



Supervised learning v.s. Un-supervised learning
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/e0285eb0-3184-44cd-8743-3ba97466354c)

Supervised learning

    概念：

        看到什麼盤勢，就落子在哪一個位置

    缺點：

        不足的地方是，有時候人也不一定知道正確答案

        棋譜上面的應對不見得是最 optimal

    舉例來說：

        Supervised learning 就是 machine 從一個老師那邊學，老師會告訴它說，每次看到這樣子的盤勢，你要下在甚麼樣的位置

Reinforcement Learning (Un-supervised)

    舉例來說：

        讓機器呢，就不管它，它就找某一個人去跟它下圍棋，從經驗中學習。

        在這幾百步裡面，甚麼樣的下法是好的，哪幾步是不好的，它要自己想辦法去知道。

    缺點：

        需要大量的 training 的 examples（可能要下三千萬盤以後，它才能夠變得很厲害）

AlphaGo 的解法

    先做 Supervised learning，訓練一批學得不錯的machine

    再讓任兩個 machine它們自己互下，去做 Reinforcement Learning。



Chat-bot
    
    chat-bot 是怎麼做的（Supervised）：

        learn一個sequence-to-sequence model

        input 是一句話，output 就是機器人回答

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/2da6af8a-5904-4e4c-aa63-b0216dfff972)

Reinforcement Learning 的 learn 法

    概念：

        讓 machine 胡亂去跟人講話，講一講以後，人最後就生氣了

        Machine 要自己去想辦法發覺說，它某句話可能講得不太好

        用AlphaGo一樣的train法，任兩個 Agent，讓它們互講。

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/4a29f493-4285-4191-8690-9839b7614fa7)

    缺點：

        對話完以後，沒有人去告訴它說，講的好還是不好（不像圍棋有明確的輸贏）

        尚待克服

    宏毅老師預言，接下來會有人用 GAN 來 learn chat-bot

        discriminator 看真正的人的對話 和 那兩個 machine 的對話，判斷這兩個 Agent 的對話像不像人

        用 discriminator 自動 learn 出給 reward 的方式


更多其他應用

    特別適合的應用就是:

        人也不知道怎麼做，你人不知道怎麼做 就代表著 沒有 labeled data
        
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/49886b3f-216c-4171-ab26-9ff26426a250)

例子：
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/98c3ee6a-57ae-4755-ae88-7eee105ab053)

Interactive retrieval：

    有一個搜尋系統，Machine 跟它說想要找尋一個跟 US President 有關的事情

    Machine 可能覺得說，這 US President 太廢了，反問它一個問題（你要找的是不是跟選舉有關的事情等等），要求它 modify

    用 Reinforcement Learning 的方式，讓 machine 學說，問甚麼樣的問題，可以得到最高的 reward。

        最後搜尋的結果，使用者覺得越好，就是 reward 越高。

        每問一個問題，對人來說，就是 extra 的 effort，可以定義為 negative reward

無人機，無人車

幫 Google 的 server 節電

造句子：summarization or translation

    因為 translation 有很多種，machine 產生出來的句子，可能是好的，卻跟答案不一樣。



Video Game (打電玩)
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/0d4d9ce9-e074-4ff0-a23c-639a480e92ea)


現成 environment (by Open AI)

    Gym

        比較舊

    Universe

        很多 3D 的遊戲
        
    那些已經內建的 AI 遊戲不同，要讓 machine 用 Reinforcement Learning 的方法，去學玩遊戲

        machine 學怎麼玩這個遊戲，其實是跟人一樣的

        看到的東西就是螢幕畫面 (pixel)，並不是從那個程式裡面去擷取甚麼東西出來

        看到這個畫面，要 take 哪個 action (要做甚麼事情)
舉例：

    Space invader (小蜜蜂)
    
    可以 take 的 action 有三個，就是左、右移動，跟開火

    Scenario



![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/ae530aee-8a06-4c4a-a169-1b29d6839b15)

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3abcbb0a-13df-4826-b573-b5de2e611e8e)

    首先machine 會看到一個 observation (s1) 以後，它要決定要 take 哪一個 action

    現在只有三個 action 可以選擇，比如說它決定要 "往右移"，會得到一個 reward

    reward 是左上角的這個分數，那往右移不會得到任何的 reward

    take 完這個 action 以後，它的 action 會影響了環境，machine 看到的 observation 就不一樣 (變成s2)

    因為它自己往右移了，但也有環境的隨機變化（比如這些外星人，也會稍微往下移動一點，或是突然多一個子彈）

    然後 machine 看到 s2 以後他要決定 take 哪一個 action，決定要射擊

    假設他成功殺了一隻外星人，他就會得到一個 reward

    他看到的 observation 就變少了一隻外星人，這個是第三個 observation (s3)

    這個 process 就一直進行下去，直到某一天在第 T 個回合的時候，他被殺死了，就進入 terminal state

    遊戲的開始到結束叫做一個 episode

![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/34bee790-68a5-4533-94e6-11de05ef2780)

對 machine 來說它要做的事情就是要不斷去玩這個遊戲，要學習在怎麼在一個 episode 裡面 maximize 它可以得到的 reward


Reinforcement Learning 的難點
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/3f912e87-eb6b-4214-964c-7e2a4e0959c1)

Reward delay (reward 的出現往往會有 delay)

    比如說在 Space Invader 裡面，只有開火這件事情才可能得到 reward

        但是如果 machine 只知道開火以後就得到 reward，它最後 learn 出來的結果它只會瘋狂開火

        對它來說往左移、往右移沒有任何 reward 它不想做

        實際上往左移、往右移這些 moving，對開火能不能夠得到 reward 這件事情是有很關鍵的影響

    machine 需要有這種遠見 (vision)，才能夠把這些電玩完好

    就跟下圍棋也是一樣，有時候短期的犧牲最後可以換來最後比較好的結果


Agent 採取的行為，會影響它之後所看到的東西

    agent 要學會去探索這個世界

    比如說在 Space Invader，如果 agent 只會往左移、往右移，從來不開火，他就永遠不知道開火可以得到 reward

        或是它從來沒有試著去擊殺最上面這個紫色的母艦，它就永遠不知道擊殺那個東西可以得到很高的 reward
        
    要讓 machine 知道要去 explore 它沒有做過的行為

Reinforcement Learning 的方法

Markov Decision Process

    暫不提

Deep Q Network

    已經被A3C完勝

A3C

    gym 裡面最強的那些 agent 都是用這個model
 
主要分為：

    Policy-based

        會 learn 一個負責做事的 actor

    Value-based

        會 learn 一個不做事的 critic，專門批評
 
把 actor 跟 critic 加起來的方法，叫做 Actor-Critic

現在最強的方法就是 Asynchronous Advantage Actor-Critic，縮寫叫 A3C

問題：最強的 Alpha Go 是用甚麼方法

    各種方法大雜燴

    有 Policy Gradient 方法、Policy-based 方法

    有 Value-based 的方法

    有 Model-based 的方法（未提到，指一個 Monte Carlo tree search 那一段）
![image](https://github.com/joycelai140420/MachineLearning/assets/167413809/1607e09c-e270-4baf-9f06-67c75f2970a4)















