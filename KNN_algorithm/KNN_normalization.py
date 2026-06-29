"""
KNN_normalization.py

我们在进行归一化处理的时候
    1. 导包
    2. 准备数据集
    3. 数据归一化处理
    4. 创建模型对象
    5. 模型训练
    6. 模型预测
"""

# 导包好的模型对象
from sklearn.preprocessing import MinMaxScaler

# 准备数据集
# 线性代数视角：这是一个 3 行 4 列矩阵，即 3 个样本、4 个特征。
x_train = [[9, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]
y_train = [0.1, 0.2, 0.3]

# 创建归一化对象
# feature_range表示映射的区间
transfer = MinMaxScaler(feature_range=(0, 1))

# 进行归一化处理
# MinMaxScaler 对每一列分别做平移和缩放，相当于改变各坐标轴的单位。
x_train_transformed = transfer.fit_transform(x_train)

# 打印归一化结果
print(x_train_transformed)
