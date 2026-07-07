"""
线性回归：正规方程示例

这一节把最小二乘的几何条件和正规方程连起来：
    X^T (y - Xw) = 0
    X^T X w = X^T y

为了让偏置也进入统一公式，这里使用设计矩阵 X_design。

对应 Markdown：
    - 线性回归第三课_正规方程.md：第 3 节 几何意义：投影到列空间
    - 线性回归第三课_正规方程.md：第 4 节 正规方程的推导结果
    - 线性回归第三课_正规方程.md：第 6 节 一元线性回归里的样子
"""

import numpy as np


# 一元线性回归数据：学习时间 -> 分数，对应线性回归第三课第 6 节。
X = np.array(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ]
)
y = np.array([35.0, 45.0, 54.0, 65.0, 74.0])

# 增加一列全 1，把偏置 b 合并进参数向量 theta = [b, w]。
# 对应第 6 节：y_hat = X_design theta，同时求截距 b 和斜率 w。
ones = np.ones((X.shape[0], 1))
X_design = np.hstack([ones, X])

print("X_design:")
print(X_design)
print()

# 1. 计算正规方程中的左右两边，对应第 4 节：X^T X theta = X^T y。
left = X_design.T @ X_design
right = X_design.T @ y

print("X_design.T @ X_design:")
print(left)
print()

print("X_design.T @ y:")
print(right)
print()

# 2. 用正规方程求解 theta：这里用 solve 解线性方程，避免显式写矩阵逆。
theta = np.linalg.solve(left, right)
y_pred = X_design @ theta
residual = y - y_pred

print("theta = [b, w]:", np.round(theta, 4))
print(f"模型公式: y_hat = {theta[1]:.2f} * x + {theta[0]:.2f}")
print()

print("预测值:", np.round(y_pred, 2))
print("残差:", np.round(residual, 2))
print()

# 3. 验证正交条件，对应第 3 节：最优残差满足 X^T(y - Xw_hat) = 0。
orthogonality = X_design.T @ residual
print("X_design.T @ residual:")
print(np.round(orthogonality, 10))
print()

# 4. 计算损失，对应线性回归第二课第 7 节：SSE = ||y - Xw||_2^2。
sse = np.sum(residual ** 2)
mse = np.mean(residual ** 2)
print(f"SSE: {sse:.2f}")
print(f"MSE: {mse:.2f}")
