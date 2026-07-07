"""
线性回归损失函数示例

这一节把“预测值 -> 残差 -> SSE / MSE / RMSE / MAE”串起来。
重点不是调用库，而是看清楚每一步量是怎么从数据中算出来的。

对应 Markdown：
    - 线性回归第二课_损失函数.md：第 2 节 预测值和残差
    - 线性回归第二课_损失函数.md：第 4-5 节 常见误差量和回归指标
    - 线性回归第二课_损失函数.md：第 7 节 线性代数写法
"""

import numpy as np


# 真实值和预测值，对应线性回归第二课第 9 节中的直观例子。
y_true = np.array([35.0, 45.0, 54.0, 65.0, 74.0])
y_pred = np.array([35.0, 44.8, 54.6, 64.4, 74.2])

# 1. 残差向量，对应第 2 节：residual = y - y_hat。
residual = y_true - y_pred

# 2. 单个样本的绝对误差和平方误差，对应第 4 节：用绝对值或平方避免正负抵消。
absolute_error = np.abs(residual)
squared_error = residual ** 2

# 3. 汇总指标，对应第 5 节：SSE、MSE、RMSE、MAE 是回归常见评价指标。
sse = np.sum(squared_error)
mse = np.mean(squared_error)
rmse = np.sqrt(mse)
mae = np.mean(absolute_error)

print("真实值 y_true:", y_true)
print("预测值 y_pred:", y_pred)
print()

print("残差 residual = y_true - y_pred:", np.round(residual, 2))
print("绝对误差 |residual|:", np.round(absolute_error, 2))
print("平方误差 residual^2:", np.round(squared_error, 2))
print()

print(f"SSE  = {sse:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")

# 对应第 7 节线性代数写法：SSE 就是残差向量二范数的平方。
sse_from_norm = np.linalg.norm(residual, ord=2) ** 2
print()
print(f"||residual||_2^2 = {sse_from_norm:.2f}")
