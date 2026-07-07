"""
线性回归第二个角度：用 NumPy 手算预测值、残差、MSE 和最小二乘解。

这一节不依赖 sklearn 的 LinearRegression，
目的是把公式 y_hat = X @ theta 和最小二乘联系起来。

线性代数主线：
    1. 给 X 增加一列全 1，用来表示偏置 b。
    2. 参数 theta = [b, w]。
    3. 预测 y_hat = X_design @ theta。
    4. 残差 residual = y - y_hat。
    5. 最小二乘寻找让 residual 的平方和最小的 theta。

对应 Markdown：
    - 线性回归第一课.md：第 5 节 线性代数写法 y_hat = Xw
    - 线性回归第二课_损失函数.md：第 7 节 SSE = ||y - Xw||_2^2
    - 线性回归第三课_正规方程.md：第 3-4 节 正交条件与正规方程
"""

import numpy as np


# 1. 准备一个简单的一元线性回归数据集，对应线性回归第一课第 3 节。
# x 表示学习时间，y 表示考试分数。
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

print("原始 X shape:", X.shape)
print("y shape:", y.shape)

# 2. 给 X 增加一列全 1，用于表示偏置 b，对应线性回归第一课第 5 节。
# 原公式：y_hat = wx + b
# 矩阵公式：y_hat = X_design @ theta
# 其中 theta = [b, w]
ones = np.ones((X.shape[0], 1))
X_design = np.hstack([ones, X])

print("增加偏置列后的 X_design:")
print(X_design)
print("X_design shape:", X_design.shape)

# 3. 先手动假设一组参数，观察预测值和误差，对应线性回归第二课第 2 节。
theta_guess = np.array([25.0, 10.0])
y_pred_guess = X_design @ theta_guess
residual_guess = y - y_pred_guess
mse_guess = np.mean(residual_guess ** 2)

print("\n手动假设 theta = [b, w]:", theta_guess)
print("预测值:", y_pred_guess)
print("残差 y - y_hat:", residual_guess)
print(f"MSE: {mse_guess:.2f}")

# 4. 使用 NumPy 的最小二乘求解，对应线性回归第三课第 7-8 节。
# np.linalg.lstsq 会寻找让 ||y - X_design @ theta||_2 最小的 theta。
theta_best, residuals, rank, singular_values = np.linalg.lstsq(
    X_design,
    y,
    rcond=None,
)

y_pred_best = X_design @ theta_best
residual_best = y - y_pred_best
mse_best = np.mean(residual_best ** 2)

print("\n最小二乘求出的 theta = [b, w]:", np.round(theta_best, 4))
print(f"模型公式: y_hat = {theta_best[1]:.2f} * x + {theta_best[0]:.2f}")
print("最优预测值:", np.round(y_pred_best, 2))
print("最优残差:", np.round(residual_best, 2))
print(f"最优 MSE: {mse_best:.2f}")

# 5. 验证最小二乘的正交条件，对应线性回归第三课第 3 节。
# 最优残差应该与 X_design 的每一列正交，即 X_design.T @ residual_best ≈ 0。
orthogonality_check = X_design.T @ residual_best

print("\n正交条件 X_design.T @ residual:")
print(np.round(orthogonality_check, 10))

# 6. 用正规方程求同样的结果，对应线性回归第三课第 4 节。
# theta = (X^T X)^(-1) X^T y
# 学习阶段可以先理解这个公式，但实际数值计算不推荐手动求逆。
theta_normal_equation = (
    np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
)

print("\n正规方程求出的 theta:")
print(np.round(theta_normal_equation, 4))
