"""
KNN处理回归问题

对应 Markdown：
    - KNN算法.md：第二节 1. 分类问题和回归问题
    - KNN算法.md：第二节 4. 回归问题：计算均值
    - KNN算法.md：第五节 2. KNN 回归与线性回归不是同一模型

关键步骤和处理分类问题大同小异
主要是有
    导包
    新建数据集
    创建模型对象
    模型训练
    模型预测
"""

# 导包好的模型对象
from sklearn.neighbors import KNeighborsRegressor

# 新建数据集，对应 KNN算法.md 第二节 1：回归问题的 y 是连续数值标签。
# x_train 仍然是特征矩阵；这里每一行是一个三维样本向量。
x_train = [[0, 0, 1],
           [1, 1, 0],
           [3, 10, 10],
           [4, 11, 12]]
y_train = [0.1, 0.2, 0.3, 0.4]

# 新建模型对象
# 对应 KNN算法.md 第一节 2：n_neighbors=3 表示预测时参考最近的 3 个邻居。
estimator = KNeighborsRegressor(n_neighbors=3)

# 模型训练
estimator.fit(x_train, y_train)

# 模型预测
# 对应 KNN算法.md 第二节 4：KNN 回归不是投票，而是对最近邻的目标值取平均。
# KNN 回归在特征空间中找最近邻并对邻居标签取平均。
# 它不同于线性回归：线性回归会求全局权重 w，并用 X @ w 预测。
print(estimator.predict([[3, 11, 10]]))
