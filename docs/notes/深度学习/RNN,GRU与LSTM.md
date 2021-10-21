# RNN, GRU 与 LSTM

## RNN

循环神经网络 (Recurrent Neural Network, RNN) 基于这样一个想法：**神经网络应该是要有一定记忆的**。例如视频的前一帧和当前帧就存在着关联，在处理句子信息时，如果能合理利用之前字词的特征，会对模型的性能有很大的提升。

### 结构

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/RNN-unrolled.png" width="50%">
</div>

顾名思义，RNN 是一种**循环**的神经网络，上图左侧并不是一个单一的节点，而是一块神经网络，$x_{t}$ / $x^{<t>}$ 是序列的 $t$ 位置（也可以理解为时间步）的输入。

将这个循环操作展开，就得到了右边部分，这样看起来就直观多了，每一个时间步隐层的均接收受前一个时间步隐层的状态作为输入。

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/architecture-rnn-ltr.png" width="60%">
</div>


形式化地，定义第 $t$ 个时间步的隐层状态 $ a^t $ / $ a^{<t>} $，输入 $x^{<t>}$，输出 $y^{<t>}$，上一隐层状态到下一隐层状态的转换权值矩阵 $W_{aa}$，输入层到隐层的权值矩阵 $W_{ax}$，隐层偏置 $ b_{a} $，隐层到输出层偏置 $b_{y}$，隐层激活函数 $g_1$，输出层激活函数 $g_2$ 。

对于每一个时间步 $t$ ，隐层状态可以表示为：
$$
a^{<t>}=g_{1}\left(W_{a a} a^{<t-1>}+W_{a x} x^{<t>}+b_{a}\right)
$$
隐层偏置 $b_a$ 可以展开为输入层到隐层的偏置 $b_{ax}$ 和隐层到隐层的偏置 $b_{aa}$：
$$
b_a = b_{aa}+b_{ax}
$$

输出可表示为：
$$
y^{<t>}=g_{2}\left(W_{y a} a^{<t>}+b_{y}\right)
$$

通常来说 $g_1$ 被设置为 tanh。

### 损失函数

在 RNN 中，损失函数被定义为每个时间步损失的和：
$$
\mathcal{L}(\hat{y}, y)=\sum_{t=1}^{T_{y}} \mathcal{L}\left(\widehat{y}^{<t>}, y^{<t>}\right)
$$

### 优势

1. 可以处理任何长度的序列
2. 模型大小不随输入序列大小而改变
3. 融合了历史（之前序列的）信息
4. 权重随时间共享

### 缺陷

1. 计算速度慢
2. 对于较长的序列，记忆效果可能会衰退
3. 不能融合未来（之后序列的）信息

### 应用

| RNN类型                           |                            示意图                            | 例子               |
| :-------------------------------- | :----------------------------------------------------------: | :----------------- |
| 一对一<br>$T_x=T_y=1$             | <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/rnn-one-to-one-ltr.png" width="80%"> | 退化为传统神经网络 |
| 一对多<br> $T_x=1, T_y>1$         | <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/rnn-one-to-many-ltr.png" width="80%"> | 音乐生成           |
| 多对一<br>$T_x>1, T_y=1 $         | <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/rnn-many-to-one-ltr.png" width="80%"> | 序列分类           |
| 多对多(等长)<br> $T_x=T_y$        | <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/rnn-many-to-many-same-ltr.png" width="80%"> | 命名实体识别       |
| 多对多（非等长）<br>$T_x\neq T_y$ | <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/rnn-many-to-many-different-ltr.png" width="80%"> | 机器翻译           |

## LSTM

在训练 RNN 的过程中，经常会遇到梯度爆炸 / 梯度消失的问题，同时，如果序列过长，那么较为靠前的信息就很难有效传递到后方。**增加门 (Gate)以控制信息的流动**可以较好地解决该问题。

长短时神经网络 (Long Short-term Memory Networks, LSTM) 在原来 RNN 的基础上增加了遗忘门，输入门和输出门。

为了使用门结构，LSTM 首先在 RNN 的基础上对隐含层 $ h_t $ 的更新方式做了一些改变：

$$
u_t=\text{tanh}(W^{xh}x_{t}+b^{xh}+W^{hh}h_{t-1}+b^{hh}) \\
$$

$$
h_{t}=h_{t-1}+u_{t}
$$

这样，我们就能对原状态的隐层参数 $h_{t-1}$ 和新状态的隐层参数 $u_{t}$ 分别赋权了，换句话说，$u_t$ 即是原来 RNN 的 $a^{t}$ 。

### <span id="forget">遗忘门</span>

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/LSTM3-focus-f.png" width="40%">
</div>

形式化地，我们可以表示遗忘门为：

$$
f_t = \sigma(W^{f,xh}x_t + b^{f,xh} + W^{f,hh}h_{t-1} + b^{f,hh})
$$

所谓门，直白地说就是一个权重，我们通常期待这个权重在 0-1 之间，所以在外层包裹了一个 sigmoid 函数。

在这个门中，参数 $W^{f,xh}$, $W^{f,hh}$, $b^{f,xh}$, $b^{f,hh}$ 都是可训练的，可以这么理解，门通过观察输入 $x_t$ 和上一隐层状态 $h_{t-1}$ 来决定需要遗忘掉多少信息。当门对应的值较小时，旧状态 $h_{t-1}$ 对当前状态的贡献也较小，也就是忘得多。

有了遗忘门，暂时我们的隐层更新公式就可以这么表示：

$$
h_t = f_t * h_{t-1} + (1-f_t) * u_t
$$

公式中 $*$ 表示元素积，即 Hadamard product。

但这就有个问题，按照这个隐层更新法则，旧状态的传递与新状态的更新是互斥的，旧状态忘得多，新状态一定更新地多，但是有的时候这两种状态对当时状态的贡献有可能同时大或者同时小，这样的更新公式就不适用了。

所以我们引入一个新的门来单独控制输入。

### 输入门

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/LSTM3-focus-i.png" width="40%">
</div>

现在就很轻车熟路了，这个门同样需要观察输入 $x_t$ 和上一隐层状态 $h_{t-1}$ 来决定新状态更新的权重。与遗忘门相比，就是换了一套可训练的参数，其他完全一致。

$$
i_t = \sigma(W^{i,xh}x_t + b^{i,xh} + W^{i,hh}h_{t-1} + b^{i,hh})
$$

有了单独对于新状态更新的控制，这样我们就得到了新的的隐层更新公式：

$$
h_t = f_t * h_{t-1} + \pmb{i_t} * u_t
$$

### 输出门

原作者在前两个门的基础上还设计了另一个门：输出门。

设计输出门的意图是，隐层信息对于该状态的输出也不一定有全部的权重，可以用一个门加以控制。

与之前一样，设计输出门为
$$
o_t = \sigma(W^{o,xh}x_t + b^{o,xh} + W^{o,hh}h_{t-1} + b^{o,hh})
$$
在 LSTM 中，我们又把未经过输出门的隐层称为记忆细胞 (Memory Cell)，把隐层更新公式中的 $ h_{t} $ 换为 $ c_{t} $，然后让其通过输出门，获得最终的权重信息：

$$
c_t = f_t * c_{t-1} + i_t * u_t
$$

$$
h_t = o_t * tanh(c_t)
$$

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/LSTM3-chain.png" width="70%">
</div>


所以，整个 LSTM 的计算图长这样。

## GRU

随着门循环单元 (Gate Recurrent Unit, GRU) 的逐渐流行，我们可以重新审视一下 LSTM，在 LSTM 的[遗忘门](#遗忘门)中提到了一个有一定缺陷的隐层更新公式，但这正是 GRU 对 RNN 进行改造的地方。

GRU 与 LSTM 相比，GRU 没有输出门，也因此没有了 Cell 的概念，这里表格中我们方便比较，还是使用了 Cell 的符号标记。同时，将隐层更新公式改为了互斥型的，在这种情况下，所谓的遗忘门和输入门被整合成了更新门，整个计算图更为简洁高效。


| 符号  |              **Gated Recurrent Unit** (GRU)              |            **Long Short-Term Memory** (LSTM)             |
| :---: | :------------------------------------------------------: | :------------------------------------------------------: |
| $u_t$ | $ \text{tanh}(W^{xh}x_{t}+b^{xh}+W^{hh}h_{t-1}+b^{hh}) $ | $ \text{tanh}(W^{xh}x_{t}+b^{xh}+W^{hh}h_{t-1}+b^{hh}) $ |
| $c_t$ | $ f_t*h_{t-1} + (1-f_t)*u_t $ （这里 $f_t$ 代表更新门 ） |                  $f_t*h_{t-1}+i_t*u_t$                   |
| $h_t$ |                   $ \text{tanh}(c_t) $                   |                 $ o_t*\text{tanh}(c_t) $                 |

## TODO

- Backward Pass of RNN
- Variants of RNN and LSTM
- RNN and LSTM in PyTorch

## 参考

https://colah.github.io/posts/2015-08-Understanding-LSTMs/

https://zhuanlan.zhihu.com/p/30844905

https://zhuanlan.zhihu.com/p/101322965

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

车万翔，郭江《自然语言处理-基于预训练模型的方法》[M]

