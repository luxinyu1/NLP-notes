# PyTorch 与 Numpy 中的乘法

乘法运算是 ML / DL 中最为基础的运算之一，两大基础框架 PyTorch 与 Numpy 提供了种类繁多的函数，然而由于中英文概念之间的差异，框架实现之间的差异，导致了使用时极易混淆，往往导致了运算结果出错或者出现维度不匹配的情况。

##  数学概念

### 点积 / 内积 / 数量积 / 元素积 (dot product)

一般地，对于空间中的两个 n 维向量 $\vec{a}, \vec{b}$  其内积的代数定义如下：
$$
a\cdot b = \sum\nolimits_{i=1}^{n}a_ib_i
$$

两个向量内积的结果是个**标量**。

### 矩阵乘积 (matmul product)

数学意义上，矩阵乘积仅在第一个矩阵的列数 (column) 和第二个矩阵的行数 (row) 相同时才有意义；

设 $A$ 是形状为 $m \times p$ 的矩阵，$ B $ 是形状为 $ p \times n$ 的矩阵，则矩阵乘积 $AB$ 第 $i$ 行第 $j$ 列的元素为：
$$
(AB)_{ij} = \sum\nolimits_{k=1}^{p}a_{ik}b_{kj}
$$

> 两个相同维数的向量 $x$ 和 $y$ 的点积可看作是矩阵乘积 $x^\mathrm{T}y$。

### 矩阵元素积 (element-wise product / Hadamard product)

矩阵 $A$ 与矩阵 $B$ 必须拥有相同的形状，类似地矩阵元素积 $A*B$ 第 $i$ 行第 $j$ 列的元素为：
$$
(A*B)_{ij} = a_{ij}b_{ij}
$$

## 广播机制

介绍乘法之前，首先应该熟悉 PyTorch 与 Numpy 中都存在的广播机制。

通常只在对多个张量进行对应元素操作形状不同时，会发生广播。广播机制可以显著减少与拷贝张量有关的代码，使算法实现更高效。

Numpy 中对于广播的官方解释：

> 将两个张量的形状 (shape) 尾对尾对齐，从最尾端的维度向前检查，满足以下两个条件之一，广播即可进行：
>
> 1. 两个张量在该维度的大小相等
> 2. 两个张量在该维度有一个大小为1

更直接地，可以举几个例子：

```
shape(8,1,6,1)
  shape(7,1,5) 
--------------
shape(8,7,6,5) // 广播结果
==============
shape(3,3)
shape(3,1)
----------
shape(3,3) // 广播结果
==========
shape(4,3)
  shape(4) 
----------
ValueError('frames are not aligned') // 不能进行广播，从尾部开始第一个维度就不符合任意一个广播条件
```

从代码层面理解，与上面两条规则对应地，主要可以减少以下两种情况的代码：

1. 增加一个空维度，如 ```x[:, np.newaxis]``` 或 ```np.expand_dims()```
2. 沿一个已有的维度进行张量堆叠，如 ```np.tile()```

PyTorch 的广播机制与 Numpy 一致。

## * 运算符

在 Numpy 和 PyTorch 中 * 运算符都代表张量**点积**，分别与 ```np.multiply()``` 与 ```torch.mul()```等价。

## @ 运算符

在 Numpy 和 PyTorch 中 @ 运算符都代表张量**乘**，分别与 ```np.matmul()``` 与 ```torch.matmul()```等价。

## dot

> torch.dot(input, other, *, out=None) → Tensor

Pytorch 中的 dot 只支持两个**一维张量**之间的运算，并且这两个一维张量所含元素个数必须相同。该运算与数学意义上的点积完全一致。

```python
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)
```
> numpy.dot(a, b, out=None)

1. 当两个待运算张量都是**一维张量**时，Numpy 中的 dot 与 torch 中的 dot 等价。

2. 当两个待运算张量都是**二维张量**时，Numpy 中的 dot 即为矩阵乘积。
3. 当两个待运算张量都是**零维张量**（标量）时，Numpy 中的 dot 即为普通乘法。

当两个待运算张量维度不匹配时，Numpy 将自动进行在多出维度上求和运算，所以，Numpy 的 dot 并不完全是数学意义上的点积运算，在其实际实现意义和数学意义不相符时，应该谨慎使用。

## matmul

> torch.matmul(*input*, *other*, ***, *out=None*) → Tensor

虽然名义上是 matmul（矩阵乘）函数，但是 PyTorch 依然做了全面的扩充，以至于官方文档中整个函数的解释说明非常复杂，还是看实际例子比较好些：

```python
>>> # vector x vector
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([])
```

当两个输入张量都是向量时 matmul() 执行 dot product 操作😶

```
>>> # matrix x vector
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
```

当两个输入张量分别为矩阵和向量时，执行矩阵-向量乘运算：
$$
A \mathbf{x}=\left[\begin{array}{cccc}
a_{11} & a_{12} & \ldots & a_{1 n} \\
a_{21} & a_{22} & \ldots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m 1} & a_{m 2} & \ldots & a_{m n}
\end{array}\right]\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right]=\left[\begin{array}{c}
a_{11} x_{1}+a_{12} x_{2}+\cdots+a_{1 n} x_{n} \\
a_{21} x_{1}+a_{22} x_{2}+\cdots+a_{2 n} x_{n} \\
\vdots \\
a_{m 1} x_{1}+a_{m 2} x_{2}+\cdots+a_{m n} x_{n}
\end{array}\right]
$$

```python
>>> # batched matrix x broadcasted vector
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
```

batched 意思是 tensor1 第一维的 10 被视为 batch，然后 tensor2 对 batch 进行广播后变为 (10, 4)，实际是一个 (3, 4) 与 (4) 的矩阵-向量乘。

```python
>>> # batched matrix x batched matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
```

将 shape[0] 也就是 10 视为 batch，其余部分正常做矩阵乘。

```python
>>> # batched matrix x broadcasted matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
```
tensor2 对 batch 进行广播，然后其余部分正常做矩阵乘。

> numpy.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj, axes, axis])

Numpy 中的 matmul 与 torch 中一致。

### bmm

功能如其名，torch 为了方便批处理张量乘运算，还专门提供了 batch matrix-matrix product 函数。

> torch.bmm(input, mat2, *, deterministic=False, out=None) → Tensor

注意 bmm 这一批处理专用函数并不提供广播机制，输入的两个张量必须严格是 **3-D** 的。

形式化地，bmm 做这样的运算：

如果```input```为一个形状为 (b, n, m) 的张量，```mat2``` 是一个形状为 (b, m, p) 的张量, 那么 ```out``` 则会是一个(b, n, p) 的张量 。

也就是将 b 视为 batch 维度，其余维度进行二维张量乘。

### mv

功能如其名，torch 中专门进行 matrix-vector product 的函数。

> torch.mv(input, vec, *, out=None) → Tensor

### mm

功能如其名，torch 中专门进行 matrix multiplication 的函数。

> torch.mm(input, mat2, *, out=None) → Tensor

同样地，该函数禁止广播。

**为了提高代码的可读性和避免出错，使用专用函数而非 matmul 是值得提倡的。**


## multiply

> torch.mul(input,other, *, out=None) → Tensor

**矩阵**点乘，其运算性质与向量点乘一致，也就是**按位乘**，注意当维度不匹配时，如果符合广播规则，该函数自动对张量进行广播。

PyTorch 官网上的例子并不合适，在这种极端情况下，矩阵按位乘和矩阵乘是等价的：

```python
>>> a = torch.randn(4, 1)
>>> a
tensor([[ 1.1207],
        [-0.3137],
        [ 0.0700],
        [ 0.8378]])
>>> b = torch.randn(1, 4)
>>> b
tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
>>> torch.mul(a, b)
tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
        [-0.1614, -0.0382,  0.1645, -0.7021],
        [ 0.0360,  0.0085, -0.0367,  0.1567],
        [ 0.4312,  0.1019, -0.4394,  1.8753]])
```
换一个例子就很明了了：
```python
>>> a = torch.randn(1,4)
>>> a
tensor([[-0.4023,  1.2729,  1.4055, -1.4597]])
>>> b = torch.randn(4,1)
>>> b
tensor([[0.1225],
        [1.0653],
        [1.4184],
        [1.1043]])
>>> torch.mul(a, b)
tensor([[-0.0493,  0.1560,  0.1722, -0.1789],
        [-0.4285,  1.3561,  1.4973, -1.5550],
        [-0.5706,  1.8055,  1.9935, -2.0704],
        [-0.4442,  1.4056,  1.5520, -1.6119]])
```

显然，如果是矩阵乘 ```torch.matmul(a,b)```，会得到一个 shape 为 (1) 的标量，在按位乘时，a 的形状 (1, 4) 被广播为 (4, 4)，b 也一样，最后 (4, 4) 与 (4, 4) 按位乘，得到 (4, 4) 的张量。

> numpy.multiply(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Numpy 中的 multiply 也和 torch 中一致。

## 时效性

- PyTorch==1.9.1
- Numpy==1.21

## 参考

https://numpy.org/doc/stable/user/theory.broadcasting.html

https://pytorch.org/docs/stable/generated/torch.matmul.html

https://mathinsight.org/matrix_vector_multiplication
