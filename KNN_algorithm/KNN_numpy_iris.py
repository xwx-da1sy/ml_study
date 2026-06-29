# 导入 NumPy。使用 np 命名空间可以避免通配符导入造成名称冲突。
import numpy as np
# 导入鸢尾花的数据集
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# shape是用来展示特征矩阵的行数和列数的，形状
print(X.shape)
print(y.shape)

# type用来展示数据类型
# ndarray是numpy中的一种数据类型，表示多维数组
print(type(X))
print(type(y))
print(isinstance(X, np.ndarray))

# ndim是用来展示特征矩阵的维度的，维数
print(X.ndim)
print(y.ndim)

# size是用来查看元素数，不是样本数
print(X.size)
print(y.size)

# 提取第一个样本的全部特征
print()
print(X[0])

# 展示一整列的全部元素
# 这里：的意思就是全部的意思
# print()
# print(X[:, 0])

# 展示前5个样本
print()
print(X[:5])

print()
print(X.shape)
print(X[:, :2].shape)
print(X[:5, :2])
print(X[:, 2:4].shape)
print(X[:10, 2:])

# 线性回归会继续使用这些形状规则：
# X.shape == (样本数, 特征数)，w.shape == (特征数,)
# X @ w 会得到包含所有样本预测值的向量。
