"""Titanic 数据预处理第一版。

本脚本接在 01_load_data.py 后面使用。
当前只做最基础的数据清洗，还不做类别编码和模型训练。
这一版特意写了比较密的逐行注释，方便初学者理解每一步。
"""

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来读取 CSV、处理表格数据和缺失值。
import pandas as pd


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# data 文件夹路径：decision_tree/titanic_case/data。
DATA_DIR = BASE_DIR / "data"

# 训练集 CSV 文件路径。
TRAIN_PATH = DATA_DIR / "titanic_train.csv"

# 测试集 CSV 文件路径。

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 Titanic 训练集和测试集。"""
    # pd.read_csv() 读取训练集 CSV，返回 DataFrame。
    train_data = pd.read_csv(TRAIN_PATH)

    # pd.read_csv() 读取测试集 CSV，返回 DataFrame。
    test_data = pd.read_csv(DATA_DIR / "titanic_test.csv")

    # 返回两个 DataFrame，调用处可以分别接收。
    return train_data, test_data


def preprocess_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """对训练集和测试集做第一版基础预处理。"""
    # copy() 复制一份训练集，避免直接修改传入的原始数据。
    train_data = train_data.copy()

    # copy() 复制一份测试集，避免直接修改传入的原始数据。
    test_data = test_data.copy()

    # 测试集的 PassengerId 后面生成预测结果文件时还要用。
    # 这里先单独保存一份，避免后面删除列之后找不到。
    test_passenger_id = test_data["PassengerId"].copy()

    # PassengerId 是编号，不作为模型特征。
    # Cabin 缺失值太多，第一版先直接删除。
    drop_columns = ["PassengerId", "Cabin"]

    # drop(columns=...) 用来删除指定列。
    train_data = train_data.drop(columns=drop_columns)

    # 测试集也要删除相同的列，保证训练集和测试集特征一致。
    test_data = test_data.drop(columns=drop_columns)

    # mean() 计算训练集 Age 的平均值。
    # 注意：这里用训练集的平均值，不用测试集，避免测试集信息泄漏。
    age_mean = train_data["Age"].mean()

    # fillna() 用指定值填充缺失值。
    train_data["Age"] = train_data["Age"].fillna(age_mean)

    # 测试集 Age 也用训练集平均值填充。
    test_data["Age"] = test_data["Age"].fillna(age_mean)

    # mode() 计算众数，也就是出现次数最多的值。
    # mode() 返回的是一个 Series，所以用 [0] 取第一个众数。
    embarked_mode = train_data["Embarked"].mode()[0]

    # 用训练集 Embarked 众数填充训练集缺失值。
    train_data["Embarked"] = train_data["Embarked"].fillna(embarked_mode)

    # 测试集 Embarked 也用训练集众数填充。
    test_data["Embarked"] = test_data["Embarked"].fillna(embarked_mode)

    # median() 计算中位数。票价 Fare 可能有极端值，所以中位数通常比平均数更稳。
    fare_median = train_data["Fare"].median()

    # 训练集 Fare 如果有缺失值，就用训练集中位数填充。
    train_data["Fare"] = train_data["Fare"].fillna(fare_median)

    # 测试集 Fare 也用训练集中位数填充。
    test_data["Fare"] = test_data["Fare"].fillna(fare_median)

    # 返回预处理后的训练集、测试集，以及单独保留的测试集 PassengerId。
    return train_data, test_data, test_passenger_id


def show_result(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """打印预处理后的数据情况。"""
    # shape 查看处理后的训练集行数和列数。
    print("处理后训练集形状：", train_data.shape)

    # shape 查看处理后的测试集行数和列数。
    print("处理后测试集形状：", test_data.shape)

    # isnull().sum() 统计训练集每一列还剩多少缺失值。
    print("\n处理后训练集缺失值统计：")
    print(train_data.isnull().sum())

    # isnull().sum() 统计测试集每一列还剩多少缺失值。
    print("\n处理后测试集缺失值统计：")
    print(test_data.isnull().sum())

    # head() 查看处理后的训练集前 5 行。
    print("\n处理后训练集前 5 行：")
    print(train_data.head())


def main() -> None:
    """脚本入口函数。"""
    # 读取原始训练集和测试集。
    train_data, test_data = load_data()

    # 对数据做第一版基础预处理。
    train_data, test_data, test_passenger_id = preprocess_data(train_data, test_data)

    # 打印预处理后的结果，确认缺失值是否处理干净。
    show_result(train_data, test_data)

    # 打印单独保留的 PassengerId，确认它没有丢。
    print("\n测试集 PassengerId 已单独保留，用于后续生成预测结果。")
    print(test_passenger_id.head())


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
