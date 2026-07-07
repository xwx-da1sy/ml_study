"""
线性回归第一课：用 sklearn 完成 fit、predict 和评估。

这一节先不手写算法，重点看懂：
1. X 为什么必须是二维矩阵。
2. y 为什么是一维标签向量。
3. 训练后得到的 coef_ 和 intercept_ 是什么。
4. 回归问题不用 accuracy，而用 MAE、MSE、RMSE、R2 评估。

对应 Markdown：
    - 线性回归第一课.md：第 2-5 节 回归问题、y_hat = wx + b、y_hat = Xw
    - 线性回归第一课.md：第 10 节 sklearn 中的重要属性
    - 线性回归第一课.md：第 11 节 回归问题的评估指标
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 单特征回归：用学习时间预测考试分数，对应线性回归第一课.md 第 3 节。
# X 的形状是 (样本数, 特征数)，即使只有 1 个特征，也要写成二维矩阵。
X = np.array(
    [
        [1.0],
        [1.5],
        [2.0],
        [2.5],
        [3.0],
        [3.5],
        [4.0],
        [4.5],
        [5.0],
        [5.5],
        [6.0],
        [6.5],
    ]
)

y = np.array([35, 40, 45, 50, 54, 60, 65, 69, 74, 78, 84, 88])

# 划分训练集和测试集：训练集用于 fit 学参数，测试集用于评估未知数据表现。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=22,
)

# 对应线性回归第一课.md 第 10 节：fit 后模型会得到 coef_ 和 intercept_。
model = LinearRegression()
model.fit(X_train, y_train)

# 对应线性回归第一课.md 第 5 节：predict 本质上计算 y_hat = wx + b。
y_pred = model.predict(X_test)

# 对应线性回归第一课.md 第 11 节：回归问题不用 accuracy，而用误差类指标和 R2。
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()

print("w / coef_:", model.coef_)
print("b / intercept_:", model.intercept_)
print(f"模型公式: y_hat = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
print()

print("测试集 X:", X_test.ravel())
print("测试集真实 y:", y_test)
print("测试集预测 y:", np.round(y_pred, 2))
print()

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
