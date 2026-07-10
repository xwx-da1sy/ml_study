"""
逻辑回归：使用 NumPy 手动实现

对应 Markdown：
    - 逻辑回归学习笔记.md：逻辑回归算法原理与推导

本节重点：
    1. 理解逻辑回归的本质：在线性回归基础上套一层 sigmoid 激活，输出概率
    2. 理解为什么分类问题不能直接用 MSE 损失（sigmoid + MSE 会导致非凸优化）
    3. 理解交叉熵损失（Binary Cross-Entropy / Log Loss）的推导与直觉
    4. 手写梯度下降，观察损失下降和决策边界收敛的过程
    5. 理解标准化对梯度下降收敛速度的影响

核心步骤：
    1. 构造二分类合成数据（两类，各 25 个样本，2 个特征）
    2. 分层划分训练集和测试集（stratify=y 保证类别比例一致）
    3. 标准化特征（使梯度下降更稳定、收敛更快）
    4. 定义 sigmoid 激活函数
    5. 定义交叉熵损失函数（Binary Cross-Entropy）
    6. 推导并实现交叉熵损失的梯度
    7. 梯度下降训练循环
    8. 在测试集上评估准确率
    9. 可视化决策边界
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ========== 1. 构造合成数据 ==========
# 两个类别的数据，各 25 个样本，每个样本有 2 个特征 (x1, x2)
# 类别 0（前 25 条）分布在特征值较小的区域，类别 1（后 25 条）分布在较大区域
X = np.array([
    [0.8, 1.1],
    [1.0, 0.9],
    [1.2, 1.0],
    [0.9, 1.3],
    [1.1, 1.2],
    [1.3, 1.0],
    [1.4, 1.1],
    [1.0, 1.5],
    [1.5, 1.3],
    [1.6, 1.2],
    [1.7, 1.0],
    [1.3, 1.4],
    [1.8, 1.1],
    [1.9, 1.2],
    [1.4, 1.6],
    [1.6, 1.5],
    [1.7, 1.4],
    [1.8, 1.3],
    [2.0, 1.4],
    [1.9, 1.6],
    [1.5, 1.8],
    [1.6, 1.9],
    [2.1, 1.5],
    [2.0, 1.7],
    [2.2, 1.6],
    [2.8, 3.0],
    [3.0, 2.9],
    [3.1, 3.2],
    [3.2, 3.0],
    [3.3, 3.1],
    [3.4, 3.2],
    [3.5, 3.0],
    [3.6, 3.3],
    [3.7, 3.1],
    [3.8, 3.4],
    [3.9, 3.2],
    [4.0, 3.5],
    [4.1, 3.3],
    [4.2, 3.6],
    [4.3, 3.4],
    [4.4, 3.7],
    [4.5, 3.5],
    [4.6, 3.8],
    [4.7, 3.6],
    [4.8, 3.9],
    [4.9, 3.7],
    [5.0, 4.0],
    [5.1, 3.8],
    [5.2, 4.1],
    [5.3, 3.9]
], dtype=float)

# 标签：前 25 个为类别 0，后 25 个为类别 1（二分类问题）
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
], dtype=int)


# ========== 2. 分层划分训练集和测试集 ==========
# stratify=y：确保训练集和测试集中两个类别的比例与原始数据保持一致
# random_state=42：固定随机种子，保证每次运行结果一致（可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% 作为测试集，80% 作为训练集
    random_state=42,    # 随机种子，保证结果可复现
    stratify=y          # 分层抽样，保持类别比例
)

# 打印各数据集的形状，帮助验证维度是否正确
print("X shape:", X.shape)              # (50, 2)
print("y shape:", y.shape)              # (50,)
print("X_train shape:", X_train.shape)  # (40, 2)
print("X_test shape:", X_test.shape)    # (10, 2)
print("y_train shape:", y_train.shape)  # (40,)
print("y_test shape:", y_test.shape)    # (10,)


# ========== 3. 标准化 ==========
# 标准化公式：z = (x - μ) / σ，使每个特征的均值为 0、标准差为 1
# 注意：Scaler 只 fit 在训练集上，然后用同样的均值和标准差 transform 测试集
# 这是为了防止数据泄露（data leakage）：测试集信息不应影响训练过程
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit: 计算训练集的均值和标准差，transform: 进行标准化
X_test = scaler.transform(X_test)        # 用训练集的参数标准化测试集（不重新 fit）

# 打印标准化后的前 5 条训练数据和测试数据，检查标准化效果
print("X_train after scaling:")
print(X_train[:5])
print("X_test after scaling:")
print(X_test[:5])


# ========== 4. 构造设计矩阵 ==========
# 在最前面补一列 1，把偏置项并入参数向量 w
X_train_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_design = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print("X_train_design shape:", X_train_design.shape)
print("X_test_design shape:", X_test_design.shape)


# ========== 5. 定义 sigmoid 激活函数 ==========
def sigmoid(z):
    """
    Sigmoid 函数：将任意实数映射到 (0, 1) 区间，输出可解释为概率

    σ(z) = 1 / (1 + e^{-z})

    参数：
        z: 输入值，标量或 NumPy 数组（线性组合的结果 X @ theta）

    返回：
        与 z 同形状的数组，每个元素在 (0, 1) 之间

    性质：
        - σ(0) = 0.5（决策边界上概率正好 50%）
        - z → +∞ 时 σ(z) → 1（模型确信预测类别 1）
        - z → -∞ 时 σ(z) → 0（模型确信预测类别 0）
        - σ'(z) = σ(z) * (1 - σ(z))（导数形式简洁，梯度下降时有用）
    """
    return 1 / (1 + np.exp(-z))


# ========== 6. 初始化参数 ==========
# 由于已经把偏置项并入 w，所以 w 的长度等于设计矩阵的列数
w = np.zeros(X_train_design.shape[1])

print("initial w:", w)


# ========== 7. 先用线性回归做一个基线 ==========
# 这里调用现成 API，并且使用 MSE 观察回归结果
# fit_intercept=False 是因为偏置项已经放进设计矩阵第一列
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(X_train_design, y_train)

w_linear = linear_model.coef_
y_train_pred_linear = linear_model.predict(X_train_design)
y_test_pred_linear = linear_model.predict(X_test_design)

train_mse = mean_squared_error(y_train, y_train_pred_linear)
test_mse = mean_squared_error(y_test, y_test_pred_linear)

print("linear regression w:", w_linear)
print("linear regression train predictions:", y_train_pred_linear[:5])
print("linear regression test predictions:", y_test_pred_linear[:5])
print("linear regression train MSE:", train_mse)
print("linear regression test MSE:", test_mse)


# ========== 8. 计算线性组合 z ==========
# 先看初始化参数下的 z，再看线性回归参数下的 z
z_train_init = X_train_design @ w
z_test_init = X_test_design @ w

z_train_linear = X_train_design @ w_linear
z_test_linear = X_test_design @ w_linear

print("z_train with initial w:", z_train_init[:5])
print("z_test with initial w:", z_test_init[:5])
print("z_train with linear regression w:", z_train_linear[:5])
print("z_test with linear regression w:", z_test_linear[:5])


# =========================
# 9. 定义预测函数（概率输出）
# =========================

# 逻辑回归的预测分两步：
#   1）先算线性组合 z = X_design @ w
#   2）再把 z 通过 sigmoid 映射为概率 p
#
# 最终返回的是该样本属于类别 1 的估计概率：
#   P(y=1 | x) = σ(z) = σ(x · w)
#
# 概率 > 0.5 就判为类别 1，概率 <= 0.5 就判为类别 0
def predict_prob(X_design, w):
    """
    计算每个样本属于类别 1 的概率。

    参数：
        X_design: 设计矩阵，形状 (样本数, 特征数+1)，第一列是全 1
        w: 参数向量，形状 (特征数+1,)，w[0] 是偏置项

    返回：
        prob: 概率向量，形状 (样本数,)，每个元素 ∈ (0, 1)
    """
    z = X_design @ w
    return sigmoid(z)


# =========================
# 10. 定义损失函数（Binary Cross-Entropy）
# =========================

# 对于二分类，逻辑回归标准使用的是交叉熵损失：
#   L = -(1/m) * Σ [ y log(p) + (1-y) log(1-p) ]
#
# 其中：
#   p 是模型预测样本属于类别 1 的概率
#   y 是真实标签，只能取 0 或 1
#
# 数值稳定性处理：
#
#   eps = 1e-12 是一个极小正数（epsilon），作用是防止 log(0) 计算出 -inf。
#   因为 log(0) = -∞，当预测概率恰好为 0 或 1 时损失函数会直接崩溃（得到 nan）。
#
#   np.clip(array, min, max) 会把数组中每个元素限制在闭区间 [min, max] 内：
#     小于 min 的值 → 变成 min
#     大于 max 的值 → 变成 max
#     在区间内的值 → 保持不变
#
#   示例：
#     np.clip([-0.1, 0.3, 0.8, 1.2], 0.0, 1.0)  →  [0.0, 0.3, 0.8, 1.0]
#     np.clip([1e-20, 0.5, 1-1e-20], 1e-15, 1-1e-15)  →  [1e-15, 0.5, 1-1e-15]
#
#   这里用 np.clip(y_prob, eps, 1-eps) 将概率截断到 [1e-12, 1-1e-12]，
#   既保留了概率的语义（极接近 0 或 1），又避免了 log(0) 的数值爆炸。
def compute_loss(y_true, y_prob):
    """
    计算平均二元交叉熵损失

    参数：
        y_true: 真实标签，形状 (样本数,)
        y_prob: 预测概率，形状 (样本数,)

    返回：
        loss: 平均交叉熵损失（标量）
    """
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )
    return loss


# =========================
# 11. 定义梯度
# =========================

# 在偏置项已经并入 w 的情况下，逻辑回归的梯度可以统一写成：
#   grad = (1/m) * X_design.T @ (p - y)
#
# 其中：
#   X_design 形状是 (m, n+1)
#   w 形状是 (n+1,)
#   p = sigmoid(X_design @ w)
#   y 形状是 (m,)
def compute_gradient(X_design, y_true, w):
    """
    计算逻辑回归交叉熵损失对参数向量 w 的梯度

    参数：
        X_design: 设计矩阵，形状 (样本数, 特征数+1)
        y_true: 真实标签，形状 (样本数,)
        w: 参数向量，形状 (特征数+1,)

    返回：
        grad: 梯度向量，形状 (特征数+1,)
    """
    y_prob = predict_prob(X_design, w)
    m = len(y_true)

    grad = (X_design.T @ (y_prob - y_true)) / m
    return grad


# =========================
# 12. 根据梯度更新参数
# =========================

# 梯度下降更新公式：
#   w := w - learning_rate * grad
def update_weights(X_design, y_true, w, learning_rate):
    """
    根据当前梯度更新参数向量 w

    参数：
        X_design: 设计矩阵，形状 (样本数, 特征数+1)
        y_true: 真实标签，形状 (样本数,)
        w: 当前参数向量，形状 (特征数+1,)
        learning_rate: 学习率

    返回：
        new_w: 更新后的参数向量
    """
    grad = compute_gradient(X_design, y_true, w)
    new_w = w - learning_rate * grad
    return new_w


# =========================
# 13. 训练循环
# =========================

# 每一轮训练都做四件事：
#   1. 用当前参数计算预测概率
#   2. 计算当前损失
#   3. 计算梯度
#   4. 按梯度下降更新参数
learning_rate = 0.1
epochs = 200
loss_history = []

for epoch in range(epochs):
    y_train_prob = predict_prob(X_train_design, w)
    train_loss = compute_loss(y_train, y_train_prob)
    grad = compute_gradient(X_train_design, y_train, w)

    loss_history.append(train_loss)
    w = update_weights(X_train_design, y_train, w, learning_rate)

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"epoch={epoch:3d}, loss={train_loss:.6f}, grad={grad}, w={w}")
