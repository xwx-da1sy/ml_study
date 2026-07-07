"""
KNN_clarification.py

对应 Markdown：
    - KNN算法.md：第一节 KNN算法简介
    - KNN算法.md：第三节 KNN解决分类问题代码实现
    - KNN算法.md：第三节 2. 为什么 x_train 要写成二维结构

步骤如下：
    导包
    准备数据集（测试集和训练集）
    创建模型对象
    模型训练
    模型预测
"""

# 导包
from sklearn.neighbors import KNeighborsClassifier

# 准备数据集，对应 KNN算法.md 第三节 1：准备特征矩阵 X 和标签向量 y。
# 线性代数视角：x_train 是 4 行 1 列的特征矩阵，形状是 (样本数, 特征数)。
# 每一行是一个样本向量，每一列是一个特征。
x_train = [[0], [1], [2], [3]]
y_train = [0, 0, 1, 1]
x_test = [[5]]

# 创建模型对象
# 对应 KNN算法.md 第一节 2：n_neighbors 就是 K 值，表示参与投票的最近邻数量。
estimator = KNeighborsClassifier(n_neighbors=2)

# 模型训练
# 对应 KNN算法.md 第二节 2：fit 保存已知样本，预测时再计算未知样本到训练样本的距离。
estimator.fit(x_train, y_train)

# 模型预测
# 对应 KNN算法.md 第二节 3：分类问题使用最近 K 个邻居的类别做多数表决。
print(estimator.predict(x_test))
