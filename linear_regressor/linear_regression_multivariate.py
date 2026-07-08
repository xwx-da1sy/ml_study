"""
多元线性回归示例

特征：
    x1 -> 学习时间（小时）
    x2 -> 睡眠时间（小时）
    x3 -> 练习题数量（道）

标签：
    y  -> 考试分数

目标：
    1. 看懂 X.shape = (样本数, 特征数)
    2. 看懂 coef_ 里每个权重如何对应一个特征
    3. 看懂多元线性回归的预测公式是怎样展开的
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 每一行是一个样本：[学习时间, 睡眠时间, 练习题数量]
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
    ]
)

# 构造一个近似线性关系的标签
y = np.array([43, 47, 53, 56, 63, 66, 73, 76, 81, 85, 90, 95], dtype=float)

feature_names = ["study_hours", "sleep_hours", "practice_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=22,
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print()

print("intercept_ (b):", round(model.intercept_, 4))
print("coef_ (w):", np.round(model.coef_, 4))
print()

print("各特征对应的权重：")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:>15}: {coef:.4f}")
print()

formula = (
    f"y_hat = {model.intercept_:.2f}"
    f" + {model.coef_[0]:.2f}*x1"
    f" + {model.coef_[1]:.2f}*x2"
    f" + {model.coef_[2]:.2f}*x3"
)
print("模型公式：")
print(formula)
print()

print("测试集真实 y:", np.round(y_test, 2))
print("测试集预测 y:", np.round(y_pred, 2))
print()

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
print()

# 手动验证第一个测试样本的预测值，帮助理解“每一行样本和 w 做点积，再加 b”
sample = X_test[0]
manual_pred = model.intercept_ + np.dot(sample, model.coef_)
print("第一个测试样本:", sample)
print(f"手动计算预测值: {manual_pred:.2f}")
print(f"模型给出的预测值: {y_pred[0]:.2f}")

