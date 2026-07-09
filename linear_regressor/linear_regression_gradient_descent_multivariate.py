"""
多元线性回归：使用梯度下降求解

这个文件会按照“从零开始、一步一步补全”的方式来写。
当前阶段我们先完成最基础的准备工作：

1. 项目介绍
2. 导包
3. 创建数据集
4. 划分训练集和测试集
5. 对特征做标准化
6. 构造设计矩阵并初始化参数

后面我们会继续在这个文件里补充：

7. 定义损失函数
8. 计算梯度
9. 使用梯度下降更新参数
10. 观察损失是否下降
11. 得到最终模型公式

为什么这个例子叫“多元线性回归”：
因为每个样本不只有 1 个特征，而是有多个特征一起影响预测结果。

本例中的三个特征分别是：
    x1 -> 学习时间（小时）
    x2 -> 睡眠时间（小时）
    x3 -> 练习题数量（道）

预测目标是：
    y  -> 考试分数

从机器学习角度看：
    X 是特征矩阵
    y 是标签向量

从线性代数角度看：
    X 的每一行是一个样本
    X 的每一列是一个特征
    y 中的每个值都和 X 中同一行样本一一对应
"""


# =========================
# 1. 导包
# =========================

# NumPy 是机器学习里最常用的数值计算工具。
# 这里我们主要用它来：
# 1）创建特征矩阵 X
# 2）创建标签向量 y
# 3）后面进行向量、矩阵和梯度下降运算
import numpy as np

# train_test_split 用来把完整数据集拆分成训练集和测试集。
# 训练集用于让模型学习参数；
# 测试集用于检验模型面对“没见过的数据”时的表现。
from sklearn.model_selection import train_test_split

# StandardScaler 用来做标准化。
# 它会对每一列特征分别执行：
# 1）减去该列均值
# 2）再除以该列标准差
#
# 标准化后，训练集中的每一列通常会变成：
# 均值约等于 0
# 标准差约等于 1
from sklearn.preprocessing import StandardScaler


# =========================
# 2. 创建数据集
# =========================

# 下面的 X 是“特征矩阵”。
# 记法：X.shape = (样本数, 特征数)
#
# 每一行表示一个样本。
# 每一列表示一个特征。
#
# 当前这个数据集一共有 12 个样本，每个样本有 3 个特征：
# 第 1 列：学习时间（小时）
# 第 2 列：睡眠时间（小时）
# 第 3 列：练习题数量（道）
X = np.array(
    [
        [1.0, 6.0, 10.0],
        [1.5, 6.5, 12.0],
        [2.0, 7.0, 15.0],
        [2.5, 6.0, 18.0],
        [3.0, 7.5, 20.0],
        [3.5, 7.0, 22.0],
        [4.0, 8.0, 25.0],
        [4.5, 7.5, 28.0],
        [5.0, 8.0, 30.0],
        [5.5, 8.5, 32.0],
        [6.0, 8.0, 35.0],
        [6.5, 8.5, 38.0],
    ],
    dtype=float,
)

# 下面的 y 是“标签向量”，也就是每个样本对应的真实分数。
# 它和 X 按行一一对应：
# X 的第 1 行样本 -> y 的第 1 个分数
# X 的第 2 行样本 -> y 的第 2 个分数
# ...
y = np.array(
    [43.0, 47.0, 53.0, 56.0, 63.0, 66.0, 73.0, 76.0, 81.0, 85.0, 90.0, 95.0],
    # dtype=float 指定数据的数值类型为浮点数（Python 中即 float64）。在 NumPy/Pandas 中，它表示数组或 DataFrame 中的数据将以浮点数形式存储和运算。
    dtype=float,
)

# 给三个特征起名字，方便后面打印和解释模型系数。
feature_names = ["study_hours", "sleep_hours", "practice_count"]

# 这里先打印一下数据集的基本结构，帮助初学阶段建立“矩阵感”。
print("X 的形状（样本数, 特征数）:", X.shape)
print("y 的形状（样本数,）:", y.shape)
print()

print("第 1 个样本的特征：", X[0])
print("第 1 个样本的标签：", y[0])
print()

print("特征名称：", feature_names)
print()


# =========================
# 3. 划分训练集和测试集（8:2）
# =========================

# 为什么要划分训练集和测试集？
# 因为如果模型既在这些数据上学习，又仍然在这些同样的数据上考试，
# 那么我们就无法判断它面对新数据时是否真的学会了规律。
#
# 所以机器学习里常见做法是：
# 1）拿出一部分数据做训练集，让模型学习
# 2）再拿出一部分数据做测试集，检查模型泛化能力
#
# 这里我们使用 8:2 的比例，也就是：
# 80% 的样本作为训练集
# 20% 的样本作为测试集
#
# 当前总样本数是 12 个，
# 所以大致会分成：
# 训练集：9 个样本
# 测试集：3 个样本
#
# random_state=22 的作用：
# 固定随机划分结果，保证每次运行代码时切分结果一致，
# 这样更方便学习和调试。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=22,
)

# 打印划分后的结果，帮助理解“训练集”和“测试集”的结构仍然和原来一致。
print("训练集 X_train 的形状：", X_train.shape)
print("测试集 X_test 的形状：", X_test.shape)
print("训练集 y_train 的形状：", y_train.shape)
print("测试集 y_test 的形状：", y_test.shape)
print()

print("训练集样本数：", X_train.shape[0])
print("测试集样本数：", X_test.shape[0])
print()


# =========================
# 4. 对特征做标准化
# =========================

# 为什么这里要标准化？
# 因为我们后面要使用“梯度下降”来训练多元线性回归。
# 梯度下降对不同特征的数值尺度比较敏感：
#
# 如果某一列特征数值特别大，而另一列特别小，
# 那么损失函数在不同方向上的变化速度会差很多，
# 这会导致梯度下降：
# 1）收敛变慢
# 2）更新过程不稳定
# 3）学习率更难调
#
# 所以在线性回归里：
# - 如果用正规方程直接解，通常不一定非要标准化
# - 如果用梯度下降训练，标准化通常更推荐
#
# 注意一个非常重要的原则：
# 只能在训练集上 fit
# 然后用训练集学到的规则去 transform 训练集和测试集
#
# 不能对测试集单独 fit，否则会造成数据泄露。
scaler = StandardScaler()

# 在训练集上“学习均值和标准差”，再完成转换
X_train = scaler.fit_transform(X_train)

# 测试集只能使用训练集学到的均值和标准差进行转换
X_test = scaler.transform(X_test)

# 标准化不会改变矩阵形状，只会改变每一列的数值尺度。
print("标准化之后：")
print("X_train 的形状：", X_train.shape)
print("X_test 的形状：", X_test.shape)
print()

# 下面两行用来观察训练集标准化后的结果：
# 每一列的均值应该接近 0
# 每一列的标准差应该接近 1
print("X_train 每列的均值：", np.round(X_train.mean(axis=0), 6))
print("X_train 每列的标准差：", np.round(X_train.std(axis=0), 6))
print()


# =========================
# 5. 构造设计矩阵并初始化参数
# =========================

# 既然我们准备“完全使用矩阵式”，那就不再把偏置 b 单独写成：
#     y_hat = Xw + b
#
# 而是统一写成：
#     y_hat = X_design @ theta
#
# 这里的做法是：
# 1）在原始特征矩阵 X 的最左边补一列全 1
# 2）把偏置和所有权重合并成一个参数向量 theta
#
# 如果原来有 3 个特征：
#     [x1, x2, x3]
#
# 那么补完以后每一行就会变成：
#     [1, x1, x2, x3]
#
# 这样参数向量 theta 就可以统一写成：
#     theta = [b, w1, w2, w3]
#
# 于是矩阵乘法会自动完成：
#     y_hat = b + w1*x1 + w2*x2 + w3*x3

# 先为训练集和测试集各自准备一列全 1。
# 行数必须和对应数据集的样本数一致。

# np.ones: Numpy的一个函数，意思是创建一个全是1的数组
# 参数列表：(行数，列数)
# X_train.shape[0]: X_train.shape 返回一个元组如 (100, 3)，[0] 取第一个元素即样本数=100
train_ones = np.ones((X_train.shape[0], 1))
test_ones = np.ones((X_test.shape[0], 1))

# 把全 1 列拼到特征矩阵左边，得到设计矩阵 X_design。
#
# np.hstack: Numpy的一个函数，即水平拼接（按列方向堆叠）多个数组
# 参数列表：([数组1, 数组2])即我想要去拼接的两个数组
# X_train_design: 拼接结果，形状为 (样本数, 特征数+1)，第一列全是 1（截距项），后面是原始特征
X_train_design = np.hstack([train_ones, X_train])
X_test_design = np.hstack([test_ones, X_test])

# 现在训练集每个样本就从 3 个特征变成了 4 列：
# 第 1 列是常数 1，用来承接偏置
# 后 3 列才是标准化后的真实特征
print("X_train_design 的形状：", X_train_design.shape)
print("X_test_design 的形状：", X_test_design.shape)
print()

# 初始化参数向量 theta。
# 因为现在总列数是 4，所以 theta 也必须有 4 个参数：
# [b, w1, w2, w3]
#
# 这里我们先全部初始化为 0。
# 对线性回归来说，这是一种很常见、也很适合初学阶段检查维度的写法。
#
# np.zeros(4): 传入单个整数，创建的是长度为 4 的一维向量 [0. 0. 0. 0.]，形状为 (4,)
# 如果要创建二维矩阵，需要传入元组，如 np.zeros((3, 4)) 创建 3 行 4 列的矩阵
#
# 使用一维向量而非二维列向量做参数，不会影响后续矩阵运算和求导：
#   - X @ theta: X 是 (m,4)，theta 是 (4,)，NumPy 自动将 theta 当作列向量，结果 (m,)
#   - X.T @ 误差: 求梯度也完全正确，得到 (4,) 的梯度向量
# 一维数组是 NumPy 中表示参数向量的标准写法
theta = np.zeros(X_train_design.shape[1], dtype=float)

print("theta 的形状：", theta.shape)
print("theta 的初始值：", theta)
print()


# =========================
# 6. 构建预测函数
# =========================

# 现在我们已经准备好了两样最关键的东西：
# 1）设计矩阵 X_design
# 2）参数向量 theta
#
# 所以下一步就是定义“预测函数”。
# 在矩阵写法里，多元线性回归的预测公式统一写成：
#     y_hat = X_design @ theta
#
# 这里的 @ 是矩阵乘法符号。
# 它会自动完成：
# 每一行样本和参数向量 theta 做点积
# 从而得到每一个样本对应的预测值。
#
# 因为我们已经把偏置合并进 theta 的第 1 个位置了，
# 所以这里不需要再单独写 + b。
def predict(X_design, theta):
    """
    使用设计矩阵和参数向量，计算所有样本的预测值。

    参数说明：
    X_design:
        设计矩阵，形状是 (样本数, 特征数 + 1)。
        第 1 列是全 1，用来承接偏置项；
        后面的列是标准化后的真实特征。

    theta:
        参数向量，形状是 (特征数 + 1,)。
        写法统一为：
            [b, w1, w2, w3, ...]

    返回值：
    y_hat:
        预测值向量，形状是 (样本数,)。
        它和标签向量 y 的形状一致，后面就可以直接拿来计算误差和损失。
    """
    return X_design @ theta


# 先用当前初始化的 theta 做一次预测。
# 因为 theta 现在全是 0，所以预测结果也会全部是 0。
# 这一步的目的不是追求预测准确，而是先验证：
# 1）预测函数能正常运行
# 2）输出形状是否正确
y_train_pred = predict(X_train_design, theta)
y_test_pred = predict(X_test_design, theta)

print("使用初始 theta 进行预测：")
print("y_train_pred 的形状：", y_train_pred.shape)
print("y_test_pred 的形状：", y_test_pred.shape)
print()

print("前 3 个训练集预测值：", y_train_pred[:3])
print("前 3 个测试集预测值：", y_test_pred[:3])
print()


# =========================
# 7. 定义损失函数（SSE）
# =========================

# 你这里明确更喜欢使用 SSE（Sum of Squared Errors，平方误差和），
# 所以我们就把它作为当前版本的损失函数。
#
# SSE 的定义是：
#     SSE = Σ (y_true - y_pred)^2
#
# 也就是：
# 1）先计算真实值和预测值之间的残差
# 2）把每个残差平方
# 3）最后把所有平方误差加起来
#
# 为什么 SSE 可以作为损失函数？
# 因为它能同时满足两点：
# 1）平方以后不会出现正负抵消
# 2）误差越大，惩罚越明显
#
# 在线性回归里，我们训练模型的目标就是：
# 不断调整 theta，
# 让 SSE 尽可能变小。
def compute_sse(y_true, y_pred):
    """
    计算平方误差和 SSE。

    参数说明：
    y_true:
        真实标签向量，形状通常是 (样本数,)。

    y_pred:
        预测值向量，形状通常也是 (样本数,)。

    返回值：
    sse:
        当前参数下所有样本的平方误差和。
        数值越小，通常说明模型拟合得越好。
    """
    residual = y_true - y_pred
    return np.sum(residual ** 2)


# 先用当前初始化的参数，分别计算训练集和测试集上的 SSE。
# 由于 theta 目前全是 0，所以这两个 SSE 会比较大，
# 这正好说明“模型还没有训练”。
train_sse = compute_sse(y_train, y_train_pred)
test_sse = compute_sse(y_test, y_test_pred)

print("当前初始参数下的 SSE：")
print(f"训练集 SSE: {train_sse:.4f}")
print(f"测试集 SSE: {test_sse:.4f}")
print()


# =========================
# 8. 定义梯度函数
# =========================

# 这一部分非常重要，因为“梯度下降”这个名字里的“梯度”，
# 说的就是这里。
#
# 我们当前使用的损失函数是：
#     SSE = Σ (y_true - y_pred)^2
#
# 又因为：
#     y_pred = X_design @ theta
#
# 所以把预测公式代入损失函数后，可以写成：
#     SSE(theta) = Σ (y - X_design @ theta)^2
#
# 从线性代数角度看：
# 1）theta 是我们要学习的参数向量
# 2）SSE(theta) 是一个“参数 -> 损失”的函数
# 3）梯度就是这个函数对每个参数的偏导数组成的向量
#
# 梯度向量可以理解成：
#     “如果我现在站在当前 theta 这个位置，
#      每个参数往哪个方向改，会让损失上升最快？”
#
# 因为梯度指向“上升最快”的方向，
# 所以后面的梯度下降才会使用：
#     theta := theta - alpha * gradient
#
# 也就是：
#     不往上走，而是朝反方向走
#     从而让损失下降
#
# ------------------------------------------------------------
# 一、先看标量直觉
# ------------------------------------------------------------
#
# 如果只有一个参数 t，
# 我们会写：
#     dJ/dt
#
# 它表示：
#     t 改变一点点时，损失 J 会变化多快
#
# 如果有多个参数，比如：
#     theta = [theta_0, theta_1, theta_2, theta_3]
#
# 那么就要分别求：
#     dJ/dtheta_0
#     dJ/dtheta_1
#     dJ/dtheta_2
#     dJ/dtheta_3
#
# 再把它们排成一个向量，这个向量就是梯度。
#
# ------------------------------------------------------------
# 二、详细的向量求导推导（从损失函数到梯度公式）
# ------------------------------------------------------------
#
# 这里我们一步一步推导"为什么梯度公式长这个样子"。
# 重要的是每一步的向量形状变化，因为它决定了矩阵乘法能不能成立。
#
# ------------------------------------------------------------
# 步骤 1：写出损失函数的向量形式
# ------------------------------------------------------------
#
# SSE 的标量写法是逐个样本累加：
#     SSE = Σ_{i=1}^{m} (y_i - ŷ_i)²
#
# 向量写法（用矩阵乘法代替循环）：
#     SSE = (y_true - y_pred)^T @ (y_true - y_pred)
#          = (y - Xθ)^T @ (y - Xθ)
#
# 为什么等价？
#   对任意向量 v = [v1, v2, ..., vm]：
#       v^T @ v = v1² + v2² + ... + vm² = Σ v_i²
#   这就是"向量自身的内积等于各分量平方和"。
#
# 各向量形状：
#     y:           (m,)     —— 真实标签向量，m 个样本
#     X:           (m, n)   —— 设计矩阵，m 样本 × n 特征（含偏置列）
#     θ:           (n,)     —— 参数向量，n 个参数
#     Xθ:          (m,)     —— 预测值向量，每个样本一个预测值
#     y - Xθ:      (m,)     —— 残差向量，每个样本一个残差
#     (y - Xθ)^T:  (m,)     —— 一维向量转置后形状仍是 (m,)（NumPy 的 .T 对一维数组无效果）
#
# 所以 SSE 是一个标量（单个数字），形状为 ()。
#
# ------------------------------------------------------------
# 步骤 2：对 θ 求梯度（链式法则 + 标量对向量求导）
# ------------------------------------------------------------
#
# 把 SSE 展开：
#     SSE(θ) = (y - Xθ)^T @ (y - Xθ)
#
# 外层是对残差向量的内积（平方和），内层是线性变换 Xθ。
# 用链式法则：
#     ∂SSE/∂θ = (∂SSE/∂残差) @ (∂残差/∂θ)
#
# 第一部分：SSE 对残差向量 v = y - Xθ 求导
#     设 v = y - Xθ = [v1, v2, ..., vm]
#     SSE = v^T @ v = v1² + v2² + ... + vm²
#     对每个 v_i 求偏导：∂SSE/∂v_i = 2 * v_i
#     所以 ∂SSE/∂v = 2v = 2(y - Xθ)    ← 这是一个 (m,) 的向量
#
# 第二部分：残差 v = y - Xθ 对 θ 求导
#     v = y - Xθ，其中 y 与 θ 无关，只关心 -Xθ
#     ∂(Xθ)/∂θ = X（对线性变换 Xθ 关于 θ 求导，结果是 X 本身）
#     所以 ∂(y - Xθ)/∂θ = -X    ← 注意负号
#
# 链式法则合并（残差对 θ 的导数是一个 n×m 矩阵，需要转置）：
#     ∂SSE/∂θ = (∂v/∂θ)^T @ (∂SSE/∂v)
#             = (-X)^T @ (2 * (y - Xθ))
#             = -2 * X^T @ (y - Xθ)
#
# 令 residual = y_true - y_pred = y - Xθ，则：
#     grad = -2 * X^T @ residual
#
# ------------------------------------------------------------
# 步骤 3：逐个验证形状，确认矩阵乘法合法
# ------------------------------------------------------------
#
#     X.T.shape = (n, m)     —— X 转置：列变行，行变列
#     residual.shape = (m,)   —— 残差向量
#
#     X.T @ residual:
#         (n, m) @ (m,) → (n,)
#
#     结果形状为 (n,)，正好等于 θ 的形状！
#     这样 θ := θ - α * grad 才能做逐元素减法。
#
# 举例（当前文件，训练集有 9 个样本，设计矩阵有 4 列）：
#     X_train_design.shape      = (9, 4)
#     X_train_design.T.shape    = (4, 9)
#     residual.shape            = (9,)
#     X_train_design.T @ residual = (4, 9) @ (9,) = (4,)
#
#     得到 4 个梯度分量，分别对应：
#         dSSE/db      （偏置的偏导数）
#         dSSE/dw1     （第1个特征权重的偏导数）
#         dSSE/dw2     （第2个特征权重的偏导数）
#         dSSE/dw3     （第3个特征权重的偏导数）
#
# ------------------------------------------------------------
# 步骤 4：两种等价写法
# ------------------------------------------------------------
#
# 写法 A（残差在前，更直观）：
#     residual = y_true - X_design @ theta
#     grad = -2 * X_design^T @ residual
#
# 写法 B（合并写法）：
#     grad = -2 * X_design^T @ (y_true - X_design @ theta)
#
# 写法 C（把负号吸收进括号）：
#     grad = 2 * X_design^T @ (X_design @ theta - y_true)
#
# 三者完全等价，只是符号摆放位置不同。
# 这个文件中使用的是写法 A，因为先算残差方便调试和日志输出。
#
# ------------------------------------------------------------
# 三、为什么梯度是一个向量
# ------------------------------------------------------------
#
# 因为 theta 本身就是一个向量。
#
# 例如当前文件里：
#     theta = [b, w1, w2, w3]
#
# 那么梯度就也必须有 4 个分量，分别对应：
#     偏置 b 的导数
#     第 1 个特征权重的导数
#     第 2 个特征权重的导数
#     第 3 个特征权重的导数
#
# 所以：
#     theta.shape    = (4,)
#     gradient.shape = (4,)
#
# 这样后面才能做同形状的更新：
#     theta := theta - alpha * gradient
#
# ------------------------------------------------------------
# 四、这里为什么不除以样本数 m
# ------------------------------------------------------------
#
# 因为你当前明确选择的是 SSE，而不是 MSE。
#
# 如果损失函数写成：
#     SSE = Σ (y_true - y_pred)^2
# 那么梯度就是：
#     grad = -2 * X_design^T @ (y_true - y_pred)
#
# 如果损失函数换成 MSE：
#     MSE = (1/m) Σ (y_true - y_pred)^2
# 那么梯度里才会多出 1/m。
#
# 两者的“最优点”是一致的，
# 区别主要体现在梯度大小不同，进而影响学习率怎么调。
def compute_gradient(X_design, y_true, theta):
    """
    计算当前参数 theta 下，SSE 对 theta 的梯度向量。

    参数说明：
    X_design:
        设计矩阵，形状是 (样本数, 特征数 + 1)。

    y_true:
        真实标签向量，形状是 (样本数,)。

    theta:
        参数向量，形状是 (特征数 + 1,)。

    返回值：
    gradient:
        梯度向量，形状与 theta 完全一致。
        里面的每一个分量，都表示“当前损失对某个参数的变化有多敏感”。
    """
    # 第一步：先根据当前参数计算预测值
    y_pred = predict(X_design, theta)

    # 第二步：计算残差向量
    residual = y_true - y_pred

    # 第三步：根据 SSE 的梯度公式计算梯度
    gradient = -2 * (X_design.T @ residual)

    return gradient


# 先用当前初始化的 theta 计算一次训练集梯度。
# 因为现在参数还是全 0，所以这个梯度会告诉我们：
# “如果想让当前训练集 SSE 下降，theta 应该先往哪个方向改”
train_gradient = compute_gradient(X_train_design, y_train, theta)

print("当前初始参数下的训练集梯度：")
print(train_gradient)
print("梯度的形状：", train_gradient.shape)
print()


# =========================
# 9. 使用梯度下降更新参数
# =========================

# 现在我们已经具备了梯度下降所需的全部核心部件：
# 1）设计矩阵 X_design
# 2）参数向量 theta
# 3）预测函数 predict(...)
# 4）损失函数 compute_sse(...)
# 5）梯度函数 compute_gradient(...)
#
# 接下来就要真正开始“下降”。
#
# 梯度下降的更新公式是：
#     theta := theta - alpha * gradient
#
# 这里的 alpha 叫学习率（learning rate）。
# 它控制每次沿着负梯度方向走多大一步。
#
# 如果 alpha 太小：
#     损失会下降，但速度很慢
#
# 如果 alpha 太大：
#     可能一步跨太远，导致损失震荡甚至发散
#
# 因为我们已经对特征做了标准化，所以这里可以比较放心地选一个中等偏小的学习率。
learning_rate = 0.01
epochs = 2000


def gradient_descent_step(X_design, y_true, theta, alpha):
    """
    执行一次梯度下降参数更新。

    参数说明：
    X_design:
        设计矩阵

    y_true:
        真实标签向量

    theta:
        当前参数向量

    alpha:
        学习率，控制每次更新步长

    返回值：
    new_theta:
        更新之后的新参数向量
    """
    gradient = compute_gradient(X_design, y_true, theta)
    new_theta = theta - alpha * gradient
    return new_theta


# 为了观察训练过程，这里记录训练开始时的损失。
initial_train_pred = predict(X_train_design, theta)
initial_train_sse = compute_sse(y_train, initial_train_pred)

print("开始训练前：")
print(f"训练集 SSE: {initial_train_sse:.4f}")
print(f"学习率 alpha: {learning_rate}")
print(f"迭代轮数 epochs: {epochs}")
print()


# 进入训练循环。
# 每一轮都做三件事：
# 1）用当前 theta 计算梯度
# 2）按梯度下降公式更新 theta
# 3）观察损失是否下降
for epoch in range(epochs):
    theta = gradient_descent_step(X_train_design, y_train, theta, learning_rate)

    # 不需要每一轮都打印，否则输出会太多。
    # 这里只在关键轮次打印，方便观察“损失是不是在往下掉”。
    if epoch in [0, 1, 2, 4, 9, 19, 49, 99, 199, 499, 999, 1999]:
        current_train_pred = predict(X_train_design, theta)
        current_train_sse = compute_sse(y_train, current_train_pred)
        print(
            f"epoch={epoch + 1:4d} | "
            f"train_sse={current_train_sse:10.4f} | "
            f"theta={np.round(theta, 4)}"
        )

print()


# =========================
# 10. 训练结束后查看结果
# =========================

train_pred = predict(X_train_design, theta)
test_pred = predict(X_test_design, theta)

train_sse = compute_sse(y_train, train_pred)
test_sse = compute_sse(y_test, test_pred)

print("训练结束后：")
print("最终 theta：", np.round(theta, 4))
print(f"训练集 SSE: {train_sse:.4f}")
print(f"测试集 SSE: {test_sse:.4f}")
print()

# theta[0] 是偏置，theta[1:] 分别对应三个特征的权重。
print("模型公式（基于标准化后的特征）：")
print(
    f"y_hat = {theta[0]:.4f}"
    f" + {theta[1]:.4f}*x1_scaled"
    f" + {theta[2]:.4f}*x2_scaled"
    f" + {theta[3]:.4f}*x3_scaled"
)
