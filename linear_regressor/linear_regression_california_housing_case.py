"""
第二个线性回归完整案例：California Housing 房价预测

这个案例的目标：
用尽量简单、尽量清楚的代码，完成一次真实回归任务的完整流程。

我们会一步一步完成下面这些事情：

1. 准备并读取数据
2. 选择数值特征和标签
3. 划分训练集和测试集
4. 只用训练集的信息来处理缺失值
5. 对特征做标准化
6. 训练线性回归模型
7. 进行预测
8. 计算 SSE、MAE、MSE、RMSE 等评估指标
9. 看看模型学到的参数是什么
10. 对结果做一个简单解释

为什么这个案例很重要？
因为前面我们学到的很多知识点，
比如：
    - 多元线性回归
    - 特征矩阵 X
    - 标签向量 y
    - 损失函数
    - 误差指标
在这个文件里都会真正落地。

额外说明：
1. sklearn 官方的 fetch_california_housing() 在当前环境里下载失败，
   所以这里使用一个稳定镜像来获取同类 California Housing 数据。
2. 原始数据里有一个文本类别特征 ocean_proximity，
   但为了保持案例简单，我们当前只使用数值特征，
   不引入独热编码等更复杂的预处理。
3. 我们这里使用 sklearn 的 LinearRegression，
   这样可以把重点放在“完整流程”和“结果理解”上。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================
# 第 1 步：确定数据文件保存位置
# ==============================
# Path(__file__) 表示“当前这个 Python 文件”的路径。
# .resolve() 会得到绝对路径。
# .parent 表示它所在的文件夹，也就是 linear_regressor 文件夹。
CURRENT_DIR = Path(__file__).resolve().parent

# 我们单独创建一个 data 文件夹，用来保存这个案例的数据文件。
DATA_DIR = CURRENT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 这是原始数据在本地保存后的路径。
RAW_DATA_PATH = DATA_DIR / "housing.csv"

# 这是数据的下载地址。
# 这份数据和 California Housing 学习案例风格一致，非常适合线性回归练习。
DATA_URL = "https://cdn.jsdelivr.net/gh/ageron/handson-ml@master/datasets/housing/housing.csv"


# ==========================================
# 第 2 步：如果本地没有数据，就先下载一份
# ==========================================
# 这样做的好处是：
# 第一次运行会下载；
# 后面再次运行时，直接读取本地文件即可，不需要反复联网。
if not RAW_DATA_PATH.exists():
    print("本地还没有 housing.csv，正在下载数据集...")

    # pandas 可以直接从网络地址读取 csv。
    # 读出来以后，我们再保存到本地。
    download_df = pd.read_csv(DATA_URL)
    download_df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8")

    print(f"数据集下载完成，已保存到：{RAW_DATA_PATH}")
else:
    print(f"检测到本地数据文件，直接读取：{RAW_DATA_PATH}")


# ============================
# 第 3 步：读取原始数据
# ============================
# 现在把 csv 文件读取成 DataFrame。
# DataFrame 可以理解为“带列名的二维表格”。
housing_df = pd.read_csv(RAW_DATA_PATH)

print("\n================ 原始数据基本情况 ================")
print("原始数据前 5 行：")
print(housing_df.head())

print("\n原始数据形状（行数, 列数）：")
print(housing_df.shape)

print("\n原始数据列名：")
print(housing_df.columns.tolist())

print("\n原始数据每一列的缺失值数量：")
print(housing_df.isnull().sum())


# =======================================================
# 第 4 步：整理出线性回归要使用的特征矩阵 X 和标签向量 y
# =======================================================
# 我们想预测的是房价，所以标签 y 选择：
# median_house_value
#
# 为了和 sklearn 的 California Housing 风格尽量接近，
# 我们手动整理出下面 8 个数值特征：
#
# MedInc      = median_income
# HouseAge    = housing_median_age
# AveRooms    = total_rooms / households
# AveBedrms   = total_bedrooms / households
# Population  = population
# AveOccup    = population / households
# Latitude    = latitude
# Longitude   = longitude
#
# 这些特征全部都是数值型，适合直接送入当前的线性回归案例。
model_df = pd.DataFrame()
model_df["MedInc"] = housing_df["median_income"]
model_df["HouseAge"] = housing_df["housing_median_age"]
model_df["AveRooms"] = housing_df["total_rooms"] / housing_df["households"]
model_df["AveBedrms"] = housing_df["total_bedrooms"] / housing_df["households"]
model_df["Population"] = housing_df["population"]
model_df["AveOccup"] = housing_df["population"] / housing_df["households"]
model_df["Latitude"] = housing_df["latitude"]
model_df["Longitude"] = housing_df["longitude"]

# 这里的 X 是“特征矩阵”。
# 每一行表示一个样本。
# 每一列表示一个特征。
X = model_df

# 这里的 y 是“标签向量”。
# 每一个值表示一个样本对应的真实房价。
y = housing_df["median_house_value"]

print("\n================ 整理后的建模数据 ================")
print("特征矩阵 X 的前 5 行：")
print(X.head())

print("\nX 的形状：")
print(X.shape)

print("\ny 的形状：")
print(y.shape)

print("\n特征列名：")
print(X.columns.tolist())

print("\n整理后特征表每一列的缺失值数量：")
print(X.isnull().sum())


# ============================================
# 第 5 步：划分训练集和测试集
# ============================================
# 训练集：用来让模型学习参数
# 测试集：用来检验模型在“没见过的新数据”上的表现
#
# test_size=0.2 表示 20% 作为测试集，80% 作为训练集
# random_state=42 是为了保证每次划分结果一致，方便复现
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n================ 数据集划分结果 ================")
print(f"训练集特征形状：{X_train.shape}")
print(f"测试集特征形状：{X_test.shape}")
print(f"训练集标签形状：{y_train.shape}")
print(f"测试集标签形状：{y_test.shape}")


# ============================================
# 第 6 步：处理缺失值
# ============================================
# 这是一个非常关键的步骤。
#
# 为什么不能直接把带缺失值的数据送进模型？
# 因为大多数传统机器学习模型都不能直接处理 NaN。
#
# 这里我们使用“中位数填补”：
# 如果某一列有缺失值，就用该列的中位数去补。
#
# 为什么用中位数？
# 因为它通常比均值更不容易受到极端值影响。
#
# 非常重要：
# 我们只能在训练集上 fit，
# 然后把训练集学到的填补规则用到测试集上。
# 这样可以避免“数据泄露”。
imputer = SimpleImputer(strategy="median")

# 在训练集上学习“每一列应该用什么数去补”
X_train_imputed = imputer.fit_transform(X_train)

# 用训练集学到的规则，去处理测试集
X_test_imputed = imputer.transform(X_test)

print("\n================ 缺失值处理完成 ================")
print("训练集缺失值填补后形状：", X_train_imputed.shape)
print("测试集缺失值填补后形状：", X_test_imputed.shape)


# ============================================
# 第 7 步：对特征做标准化
# ============================================
# 标准化的目标：
# 让不同特征大致处于相近的数值尺度上。
#
# 标准化公式：
# x_standard = (x - 均值) / 标准差
#
# 为什么做标准化？
# 1. 有助于让不同特征处在相近量级
# 2. 有助于我们更稳定地理解模型参数
# 3. 对很多模型和优化方法都很有帮助
#
# 和缺失值处理一样：
# 也必须只在训练集上 fit，
# 再把同样的变换应用到测试集。
scaler = StandardScaler()

# 在训练集上学习均值和标准差
X_train_scaled = scaler.fit_transform(X_train_imputed)

# 用训练集学到的均值和标准差去变换测试集
X_test_scaled = scaler.transform(X_test_imputed)

print("\n================ 标准化完成 ================")
print("训练集标准化后前 3 行：")
print(X_train_scaled[:3])

print("\n测试集标准化后前 3 行：")
print(X_test_scaled[:3])


# ============================================
# 第 8 步：创建并训练线性回归模型
# ============================================
# LinearRegression 会自动学习：
# 1. 每个特征前面的权重 w
# 2. 一个截距 b
#
# 在数学上，它对应的形式是：
# y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
#
# 注意：
# 这里的 x1, x2, ..., xn 指的是“标准化之后”的特征。
model = LinearRegression()

# 用训练集训练模型
model.fit(X_train_scaled, y_train)

print("\n================ 模型训练完成 ================")
print("截距 b：")
print(model.intercept_)

print("\n权重向量 w：")
print(model.coef_)


# ============================================
# 第 9 步：使用模型进行预测
# ============================================
# 模型训练完成后，我们就可以把测试集送进去，
# 让模型输出“预测房价”。
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\n================ 预测完成 ================")
print("测试集前 5 个真实值：")
print(y_test.iloc[:5].tolist())

print("\n测试集前 5 个预测值：")
print(y_test_pred[:5].tolist())


# ============================================
# 第 10 步：计算回归评估指标
# ============================================
# 这里我们重点看四个指标：
#
# 1. SSE  = 平方误差和
# 2. MAE  = 平均绝对误差
# 3. MSE  = 均方误差
# 4. RMSE = 均方根误差
#
# 它们的作用：
# 都是用来衡量“预测值和真实值差得有多远”。
#
# 一般来说：
# 指标越小，说明预测误差越小，模型通常越好。

# 训练集指标
train_sse = np.sum((y_train - y_train_pred) ** 2)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# 测试集指标
test_sse = np.sum((y_test - y_test_pred) ** 2)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("\n================ 模型评估结果 ================")
print("训练集指标：")
print(f"SSE  = {train_sse:.4f}")
print(f"MAE  = {train_mae:.4f}")
print(f"MSE  = {train_mse:.4f}")
print(f"RMSE = {train_rmse:.4f}")

print("\n测试集指标：")
print(f"SSE  = {test_sse:.4f}")
print(f"MAE  = {test_mae:.4f}")
print(f"MSE  = {test_mse:.4f}")
print(f"RMSE = {test_rmse:.4f}")


# ============================================
# 第 11 步：把模型公式打印出来，帮助理解参数
# ============================================
# 这里打印的是“标准化特征空间下”的模型公式。
# 也就是说：
# 每个特征先经过标准化，
# 然后再乘上对应权重。
print("\n================ 模型公式（基于标准化后的特征） ================")

feature_names = X.columns.tolist()
equation_parts = []

for feature_name, coef_value in zip(feature_names, model.coef_):
    equation_parts.append(f"({coef_value:.4f} * {feature_name}_scaled)")

equation_text = " + ".join(equation_parts)
print(f"y_hat = {model.intercept_:.4f} + {equation_text}")


# ============================================
# 第 12 步：简单观察每个特征的权重
# ============================================
# 权重绝对值越大，通常说明这个特征对预测结果的影响越明显。
# 但这里要注意：
# “影响明显”不等于“因果关系成立”。
coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }
)

coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

print("\n================ 特征权重观察 ================")
print(coef_df)


# ============================================
# 第 13 步：给出一个简短结论
# ============================================
print("\n================ 案例小结 ================")
print("1. 我们使用了 8 个数值特征来预测房价。")
print("2. 先划分训练集/测试集，再处理缺失值和标准化，这是正确流程。")
print("3. 模型已经成功完成训练，并输出了预测结果。")
print("4. 你现在看到的 MAE、MSE、RMSE，就是线性回归中非常常见的评估指标。")
print("5. 如果后面想继续提升效果，可以尝试：")
print("   - 加入 ocean_proximity 这个类别特征")
print("   - 做更深入的特征工程")
print("   - 对比其他回归模型")
