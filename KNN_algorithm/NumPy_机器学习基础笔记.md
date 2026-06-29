# NumPy 机器学习基础笔记

## 1. 为什么机器学习里经常看到 NumPy

在机器学习中，很多数据都会被组织成 NumPy 数组，因为它适合表示向量、矩阵和批量数值计算。

例如鸢尾花案例中：

```python
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target
```

这里：

```text
x -> 特征矩阵
y -> 标签向量
```

## 2. 为什么会看到 `ndarray`

执行下面代码时：

```python
print(type(x))
print(type(y))
```

通常会看到：

```python
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
```

这不是报错，而是在说明：

```text
x 是 NumPy 的 ndarray 数组
y 也是 NumPy 的 ndarray 数组
```

`ndarray` 的全称可以理解为：

```text
n-dimensional array
多维数组
```

在鸢尾花案例中：

```text
x -> 二维 ndarray，保存特征矩阵
y -> 一维 ndarray，保存标签向量
```

## 3. `shape`、`ndim`、`size` 的区别

可以运行：

```python
print(x.ndim)
print(x.shape)
print(x.size)

print(y.ndim)
print(y.shape)
print(y.size)
```

对于鸢尾花数据，通常结果是：

```python
2
(150, 4)
600

1
(150,)
150
```

含义分别是：

```text
x.ndim  = 2        -> x 是二维数组
x.shape = (150, 4) -> x 有 150 行、4 列
x.size  = 600      -> x 一共有 600 个元素

y.ndim  = 1        -> y 是一维数组
y.shape = (150,)   -> y 有 150 个标签
y.size  = 150      -> y 一共有 150 个元素
```

这里要特别注意：

```text
shape 看的是数组形状
size 看的是元素总数
```

所以：

```text
x.shape = (150, 4)
x.size = 150 * 4 = 600
```

## 4. 特征矩阵的行和列

机器学习里通常把特征数据写成矩阵形式：

```text
X.shape = (样本数, 特征数)
```

鸢尾花数据中：

```text
x.shape = (150, 4)
```

含义是：

```text
150 个样本
每个样本 4 个特征
```

可以这样理解：

```text
每一行 = 一个样本
每一列 = 一个特征
```

在鸢尾花案例里：

```text
每一行 = 一朵花
每一列 = 一个花的特征
```

例如：

```python
print(x[0])
```

表示取第 1 朵花的全部特征，可能输出：

```python
[5.1 3.5 1.4 0.2]
```

这 4 个数分别表示：

```text
萼片长度
萼片宽度
花瓣长度
花瓣宽度
```

## 5. 二维数组索引：`x[行, 列]`

二维数组的基本访问方式是：

```python
x[行, 列]
```

注意 Python 从 `0` 开始计数。

例如：

```python
print(x[0, 0])
print(x[0, 1])
```

含义是：

```text
x[0, 0] -> 第 1 朵花的第 1 个特征
x[0, 1] -> 第 1 朵花的第 2 个特征
```

常见写法：

```python
x[0]       # 第 1 个样本的全部特征
x[0, 0]    # 第 1 个样本的第 1 个特征
x[:5]      # 前 5 个样本
x[:5, 0]   # 前 5 个样本的第 1 个特征
```

可以记住：

```text
逗号前面管行
逗号后面管列
冒号表示范围或全部
```

## 6. 切片：取某几行、某几列

在 NumPy 中，冒号 `:` 很常用。

```python
x[:, 0]
```

读作：

```text
取所有行的第 0 列
```

也就是取所有样本的第 1 个特征。

如果想取所有样本的前 2 个特征：

```python
x[:, :2]
```

含义是：

```text
所有行，前 2 列
```

它的形状是：

```python
x[:, :2].shape
```

结果：

```python
(150, 2)
```

如果想取前 5 个样本的前 2 个特征：

```python
x[:5, :2]
```

含义是：

```text
前 5 行，前 2 列
```

如果想取所有样本的后 2 个特征：

```python
x[:, 2:4]
```

或：

```python
x[:, 2:]
```

含义是：

```text
所有行，第 2 列到最后
```

在鸢尾花数据中，对应：

```text
花瓣长度
花瓣宽度
```

## 7. `x[:, 0]` 和 `x[:, 0:1]` 的区别

这两个写法很像，但结果形状不同。

```python
a = x[:, 0]
b = x[:, 0:1]

print(a.shape)
print(b.shape)
```

结果通常是：

```python
(150,)
(150, 1)
```

区别是：

```text
x[:, 0]   -> 取第 0 列，结果会变成一维数组
x[:, 0:1] -> 取第 0 列到第 1 列之前，结果仍然保持二维结构
```

也就是说：

```text
(150,)   -> 150 个数字
(150, 1) -> 150 行、1 列的特征矩阵
```

机器学习模型通常要求输入特征 `X` 是二维结构：

```text
X.shape = (样本数, 特征数)
```

所以如果只用一个特征训练模型，更推荐保留二维形式：

```python
x_one_feature = x[:, 0:1]
```

这样：

```text
x_one_feature.shape = (150, 1)
```

表示：

```text
150 个样本，每个样本 1 个特征
```

## 8. `reshape`：把一维数组改成二维数组

如果已经得到了一个一维数组：

```python
a = x[:, 0]
print(a.shape)
```

结果是：

```python
(150,)
```

可以用 `reshape` 改成二维：

```python
a = a.reshape(-1, 1)
print(a.shape)
```

结果变成：

```python
(150, 1)
```

这里：

```text
-1 表示这一维由 NumPy 自动计算
1 表示保留 1 列
```

所以：

```python
reshape(-1, 1)
```

可以理解成：

```text
把一串数字整理成 n 行 1 列
```

在机器学习里，这个写法很常见：

```python
model.fit(x[:, 0].reshape(-1, 1), y)
```

不过如果一开始就想保留二维，更简单的写法是：

```python
model.fit(x[:, 0:1], y)
```

## 9. 当前阶段要记住的 NumPy 重点

现阶段不需要一次性学完整个 NumPy，先掌握和机器学习最相关的部分：

```text
1. ndarray 是 NumPy 数组
2. shape 表示数组形状
3. ndim 表示数组维度
4. size 表示元素总数
5. X.shape = (样本数, 特征数)
6. y.shape = (样本数,)
7. X[行, 列] 用来访问二维数组
8. 冒号 : 表示全部或范围
9. X[:, 0] 会得到一维数组
10. X[:, 0:1] 会保留二维特征矩阵
```

## 10. 与线性代数和线性回归的联系

NumPy 的 `shape` 不只是代码格式，它对应矩阵乘法能否成立。

假设有 `m` 个样本、`n` 个特征：

```text
X.shape = (m, n)
w.shape = (n,)
```

那么：

```python
y_pred = X @ w
```

对应线性代数中的：

```text
y_hat = Xw
```

结果形状是：

```text
y_pred.shape = (m,)
```

这里每一个预测值，都是一个样本行向量和权重向量的点积：

```text
y_hat_i = x_i^T w
```

`X[:, 0:1]` 保持 `(m, 1)` 尤其重要：一元线性回归虽然只有一个特征，`scikit-learn` 仍要求 `X` 是“样本数 × 特征数”的二维特征矩阵。

下一阶段需要在现有 NumPy 基础上补充：

```text
1. 向量点积：x @ w
2. 矩阵乘法：X @ w
3. 转置：X.T
4. 单位矩阵：np.eye
5. 线性方程与最小二乘：np.linalg.lstsq
```

这些操作会分别出现在预测公式、正规方程和最小二乘求解中。详细联系见：[线性回归与线性代数学习地图](../linear_regressor/线性回归与线性代数学习地图.md)。
