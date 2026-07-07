"""
NumPy 与 Iris 特征矩阵基础。

对应 Markdown：
    - NumPy_机器学习基础笔记.md：第 2 节 为什么会看到 ndarray
    - NumPy_机器学习基础笔记.md：第 3 节 shape、ndim、size 的区别
    - NumPy_机器学习基础笔记.md：第 4-8 节 行列规则、切片和 reshape
    - NumPy_机器学习基础笔记.md：第 10 节 与线性代数和线性回归的联系
"""

# 导入 NumPy。使用 np 命名空间可以避免通配符导入造成名称冲突。
import numpy as np
# 导入鸢尾花的数据集
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 对应 NumPy 笔记第 3 节：shape 展示数组形状，不等于元素总数。
print(X.shape)
print(y.shape)

# type用来展示数据类型
# 对应 NumPy 笔记第 2 节：ndarray 是 NumPy 的 n-dimensional array，可表示向量/矩阵/更高维数组。
print(type(X))
print(type(y))
print(isinstance(X, np.ndarray))

# 对应 NumPy 笔记第 3 节：ndim 看数组维数；X 是二维特征矩阵，y 是一维标签向量。
print(X.ndim)
print(y.ndim)

# 对应 NumPy 笔记第 3 节：size 是元素总数，不是样本数。
print(X.size)
print(y.size)

# 对应 NumPy 笔记第 4 节：X[0] 提取第一个样本的全部特征。
print()
print(X[0])

# 展示一整列的全部元素
# 这里：的意思就是全部的意思
# print()
# print(X[:, 0])

# 展示前5个样本
print()
print(X[:5])

# 对应 NumPy 笔记第 6 节：切片格式是 X[行范围, 列范围]。
print()
print(X.shape)
print(X[:, :2].shape)
print(X[:5, :2])
print(X[:, 2:4].shape)
print(X[:10, 2:])

# 线性回归会继续使用这些形状规则：
# X.shape == (样本数, 特征数)，w.shape == (特征数,)
# X @ w 会得到包含所有样本预测值的向量。
