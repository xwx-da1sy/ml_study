"""
KNN_normalization.py

对应 Markdown：
    - KNN算法.md：第三节 6. 归一化和标准化的算法公式
    - 标准化学习笔记.md：第 5 节 从线性代数角度理解标准化
    - 标准化学习笔记.md：第 9 节 标准化和归一化的区别

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

# 准备数据集，对应标准化学习笔记.md 第 5 节：行表示样本，列表示特征。
# 线性代数视角：这是一个 3 行 4 列矩阵，即 3 个样本、4 个特征。
x_train = [[9, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]
y_train = [0.1, 0.2, 0.3]

# 创建归一化对象
# 对应 KNN算法.md 第三节 6：Min-Max 归一化把每一列缩放到 feature_range 指定区间。
transfer = MinMaxScaler(feature_range=(0, 1))

# 进行归一化处理
# 对应标准化学习笔记.md 第 9 节：归一化使用 min/max，标准化使用 mean/std。
# MinMaxScaler 对每一列分别做平移和缩放，相当于改变各坐标轴的单位。
x_train_transformed = transfer.fit_transform(x_train)

# 打印归一化结果
print(x_train_transformed)
