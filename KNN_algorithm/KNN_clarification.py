"""
KNN_clarification.py
步骤如下：
    导包
    准备数据集（测试集和训练集）
    创建模型对象
    模型训练
    模型预测
"""

# 导包
from sklearn.neighbors import KNeighborsClassifier

# 准备数据集
# 线性代数视角：x_train 是 4 行 1 列的特征矩阵。
# 每一行是一个样本向量，每一列是一个特征。
x_train = [[0], [1], [2], [3]]
y_train = [0, 0, 1, 1]
x_test = [[5]]

# 创建模型对象
# 这里的参数的意思是近邻数是2，estimator == model
estimator = KNeighborsClassifier(n_neighbors=2)

# 模型训练
# 注入特征和特征对应的标签
estimator.fit(x_train, y_train)

# 模型预测
print(estimator.predict(x_test))
