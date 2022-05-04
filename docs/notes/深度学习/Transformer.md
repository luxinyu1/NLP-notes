# Transformer

## 预备知识

### 语言表示学习

Transformer 是动态词嵌入的一种，与之对应的静态词嵌入拥有着一个很致命的问题：尽管其训练过程中依赖着上下文的词共现关系，但是在实际使用时是词与词之间是独立的，缺乏对上下文信息的感知。

#### 长序依赖

长序依赖 (Long-term Dependency) 是一种语言学现象。用通俗的话来说，也就是在一句句子中，相互依赖的词可能间隔很远。这对 RNN 一类的网络理解语言产生了很大的难度，尽管从模型结构设计上存在解决长序依赖的可能，但是因为链式法则本身的问题，间隔较远的隐层之间很难产生关联。全连接的神经网络则提供了一种解决长序依赖的方法，但是全连接层的所有权重都是固定的，不会随着输入长度的变化产生变化，这也不符合文本的特征。

#### 多义性

在上下文语境下，同样的词可以有不同的意思。

#### 上下文建模

从上面两个语言学现象中就可以看出对于理解 / 生成一个词，上下文的重要性了，有一类网络使用上下文编码器 (Contextual Encoder)，将独立的，也就是上下文无关的词向量编码为上下文相关的词向量。同时，基于卷积神经网络 (CNN-based) 与 RNN 一类的网络都提供了上下文建模的方法，但都各有比较大的缺点。

### Encoder - Decoder 架构

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/transformer_encode_decode.gif" width="70%"/>
</div>
上面的图动态地解释了 Encoder 与 Decoder 的功能，Encoder 主要关注源序列的特征提取与融合，也就是“编码”。Decoder 则根据源序列和已经生成的目标序列逐步生成目标序列，也就是“解码”。


## 注意力机制

在 Transformer 之前，已经出现了很多种不同的注意力机制。大多数都是直接使用一个输入向量同时承担查询、键和值三个角色，这样模型参数量少但是计算复杂，不利于学习，随着时间的推移，注意力机制的计算主要演变了 Q-K-V 形式，其主要利用 Query (Q), Key (K) 和 Value (V) 三个矩阵进行注意力求解。

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/attention_QKV.jpeg">
</div>

如上图所示，比如对于一个输入$ X $，其有 $N$ 个 token，将其分别乘上可学习参数 $ W_q$ ，$W_k$ $ (N \times D_k) $ 与 $W_v(N \times D_v)$ ，（PyTorch 注意力机制的实现中默认 $D_k=D_v$ ）就可以分别得到 $Q,K,V$，然后通过注意力公式计算得到每一个 token 位置的 attention score，也就是与其他所有 token 的相关度，最后将其叠加在 $V$ 上，注意这里的 embedding dim 由最初的的 $D_x$ 变为了 $D_v$，这就是单独设计价值投影矩阵 $ D_v $ 的效果，扩充了整个模型的参数量，也提高了注意力机制的表征能力。

有一个小问题经常会在面试中出现，为什么我们要将点积结果除掉 $\sqrt{D_k}$，标准答案是避免因维度过大导致点积结果过大。为什么点积结果不能过大？因为 softmax 分布的输入中一旦出现了很大的值，较小的值就会很惨，比如 $ \frac{e}{e^1+e^{10}+e^5} \approx 1e-4$ 而 $ \frac{e^{10}}{e^1+e^{10}+e^5} \approx 0.993$，整个分布就会比较尖锐。

形式化地，Q-K-V 形式的注意力计算公式如下：
$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^{T}}{\sqrt{D_k}})V
$$

## 多头自注意力机制

注意力机制在先前主要用于 Seq2Seq 任务中，对每一个时间步，取前一个时间步的输出与源语言端的每一个 token进行 attention 计算，这样就可以比较好地解决长序依赖问题。

自注意力则是该句子自己与自己做注意力运算，最后得出的注意力分布可以比较好的描述一个 token 在当前句子中的重要情况。

多头就是初始化多组 $ W_q, W_k, W_v $ 供学习 ，记为 $W_q, W_k, W_v$ 又进一步增大了参数量，通过这样的操作，可以获得**多种语义空间的**注意力分布，最后把这些注意力分布全部 concat 起来，再用一个 $W^O$ 压缩一下得到最后的多头注意力值。

$$
\text{head}_i = \text{Attention}(QW_i^{Q},KW_i^{K},VW_i^{V})
$$

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\text{head}_2,…,\text{head}_h) W^{O}
$$

## 模型细节

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/transformer_arch.PNG" width="40%">
</div>


### 位置编码

如图，位置编码被加在了 Input Embedding 上，这里的加就是直接做加法。具体地，为了解决 Attention Block 没有位置信息的问题，作者用正余弦函数建模了位置。
$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} &=\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$

### PFFN

$$
\mathrm{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$

逐位置的前馈神经网络 (Position-wise Feed-Forward Networks) 也是 Attention Block 中的一个模块，但这个具体作用是什么？有一种解释认为：给模型引入非线性，但是 Attention 计算的时候就自带一个 Softmax，这样很难解释地通。PaddlePaddle Edu 与 nndl-book 中均提到了“信息地提取与综合”这样一种观点，我认为是更为合理的，因为作者在设计网络的时候将隐层维数扩大了输入输出维度的四倍，也就是 2048。

### 残差连接

残差连接是深度学习中改变模型结构以提升效果的常用手段。具体在 Transformer 中，Encoder Block 与 Decoder Block 有实际作用的层后面都使用了残差连接：
$$
X_{hidden}=X_{embedding}+\text{Multi-Head\_Attention}(Q,K,V)
$$

$$
X_{hidden}=X_{Feed\_Forward}+X_{hidden}
$$

### 层归一化

层归一化 (Layer Normalization) 的作用是把神经网络中隐藏层归一为标准正态分布，也就是独立同分布 (i.i.d.)，以起到加快训练速度，加速收敛的作用。

具体地，先求均值和方差：
$$
\mu_{L}=\frac{1}{m} \sum_{i=1}^{m} x_{i}
$$

$$
\delta^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu\right)^{2}
$$

$$
\text{Layer\_Norm}\left(x_{i}\right)=\alpha \frac{x_{i}-\mu_{L}}{\sqrt{\delta^{2}+\epsilon}}+\beta
$$

加一个 $\epsilon$ 是为了防止分母为 0 ，各种模型的代码实现中也非常常见，类似于 epsilon=1e-5 这样。$ \alpha $ 和 $\beta$ 都是可学习的参数，观察公式我们就可以发现，这两个参数用于平移与缩放这个正态分布，一般 $ \alpha $ 初始化为 1，$ \beta $ 初始化为 0，也就是一个标准的正态分布。

### Masked Multi-Head Attention

可以注意到 Decoder 单元在解码 target 端输入的时候，用了 Mask 来遮挡当前 timestep 之后的词。原因很简单，机器翻译测试的时候是不能看到当前 timestep 后的词的。训练和测试应该保持一致。

具体的实现也非常直接，首先生成一个下三角全 0，上三角全为负无穷的矩阵，然后将其与 Scaled Multi-Head Attention Scores 相加即可，之后再做 softmax，就能将 -inf 变为 0，得到的这个矩阵即为每个 token 之间的权重。

### Shifted Right

机器翻译中的常规操作，添加形如```<s>```的开始标记与```</s>```的结束标记，使模型输出整体右移一位，方便翻译过程的开始与结束。

## 模型改进

- 针对 Transformer 拥有 ```max_length=512``` 的限制，提升其在 document-level 的能力。
- 针对下游任务对 Transformer 结构进行结构上的改进
- PTM 使用 Transformer 进行预训练，随后进行迁移学习

## 变体

### 降低 Attention 的计算量

有一类研究通过稀疏化原来 $N \times N$ 的 Attention Score 计算矩阵来提升 Transformer 的计算速度：

1. 原子级基于位置的 Sparse Attention

<div align="center">
    <img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/sparse_attention.PNG" width="80%">
</div>

2. 组合 Sparse Attention，可以看到这些都是基于原子级的组合

   <div align="center">
   	<img src="https://nlp-notes.oss-cn-beijing.aliyuncs.com/imgs/compond_sparse_attention.PNG" width="80%">
   </div>

   Star Transformer 主要采用了 Global Memory 的思想，也是一种先验知识，也就是将所有的句子中所有 token 节点的表征汇聚到一个全局节点（类似于 BERT 预训练时添加的```[SEP]```标签），Star Transformer 更适合小或者中等规模的数据。




## 参考

Video: A Tutorial of Transformer

《nndl-book》

《自然语言处理 - 基于预训练模型的方法》

https://zhuanlan.zhihu.com/p/265108616

https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html

