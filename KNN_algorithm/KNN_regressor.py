"""
KNN处理回归问题

关键步骤和处理分类问题大同小异
主要是有
    导包
    新建数据集
    创建模型对象
    模型训练
    模型预测
"""

# 导包
from sklearn.neighbors import KNeighborsRegressor

# 新建数据集
x_train = [[0, 0, 1],
           [1, 1, 0],
           [3, 10, 10],
           [4, 11, 12]]
y_train = [0.1, 0.2, 0.3, 0.4]

# 新建模型对象
estimator = KNeighborsRegressor(n_neighbors=3)

# 模型训练
estimator.fit(x_train, y_train)

# 模型预测
print(estimator.predict([[3, 11, 10]]))