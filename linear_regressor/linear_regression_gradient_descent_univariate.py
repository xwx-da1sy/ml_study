"""
一元线性回归：梯度下降示例

这一节重点观察：
    1. w 和 b 如何一步一步更新
    2. 损失函数如何随着迭代逐渐下降
    3. 梯度下降与正规方程的区别：不是一步算出，而是逐步逼近
"""

import numpy as np


# 一元线性回归数据：学习时间 -> 分数
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([35.0, 45.0, 54.0, 65.0, 74.0])

m = len(X)

# 1. 初始化参数
w = 0.0
b = 0.0

# 2. 设置学习率和迭代轮数
alpha = 0.01
epochs = 2000


def compute_loss(x, target, weight, bias):
    pred = weight * x + bias
    return np.mean((target - pred) ** 2) / 2


print("初始参数：")
print(f"w = {w:.4f}, b = {b:.4f}")
print(f"初始损失 = {compute_loss(X, y, w, b):.4f}")
print()

for epoch in range(epochs):
    y_pred = w * X + b
    error = y - y_pred

    # 对应 J(w,b) = (1/2m) Σ (y_i - w x_i - b)^2 的偏导
    dw = -(1 / m) * np.sum(X * error)
    db = -(1 / m) * np.sum(error)

    # 按负梯度方向更新
    w = w - alpha * dw
    b = b - alpha * db

    if epoch in [0, 1, 2, 4, 9, 19, 49, 99, 199, 499, 999, 1999]:
        loss = compute_loss(X, y, w, b)
        print(
            f"epoch={epoch + 1:4d} | "
            f"w={w:8.4f} | "
            f"b={b:8.4f} | "
            f"loss={loss:10.4f}"
        )

print()
print("训练结束：")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
print(f"最终损失 = {compute_loss(X, y, w, b):.4f}")
print(f"模型公式: y_hat = {w:.2f} * x + {b:.2f}")

