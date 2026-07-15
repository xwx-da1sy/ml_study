"""Titanic 特征编码第一版。

本脚本接在 02_prepare_features.py 后面使用。
目标是把字符串类型的字段转换成机器学习模型可以识别的数字特征。

当前处理策略：

1. 删除 Name 和 Ticket。
2. 将 Sex 编码成 0/1。
3. 将 Embarked 做 one-hot 编码。
4. 从训练集中分离出特征 X 和标签 y。
"""

# importlib.util 用来从指定文件路径加载 Python 模块。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来处理表格数据。
import pandas as pd


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# 第二步预处理脚本的文件路径。
PREPARE_FEATURES_PATH = BASE_DIR / "02_prepare_features.py"


def load_prepare_features_module():
    """从 02_prepare_features.py 文件中加载函数。"""
    # spec_from_file_location() 根据文件路径创建一个模块说明对象。
    spec = importlib.util.spec_from_file_location(
        "prepare_features",
        PREPARE_FEATURES_PATH,
    )

    # module_from_spec() 根据模块说明对象创建一个真正的模块对象。
    module = importlib.util.module_from_spec(spec)

    # exec_module() 执行模块文件，让里面的函数可以被调用。
    spec.loader.exec_module(module)

    # 返回加载好的模块，后面可以用 module.load_data() 调用函数。
    return module


def encode_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """对 Titanic 数据进行特征编码。"""
    # copy() 复制一份训练集，避免直接修改传入的数据。
    train_data = train_data.copy()

    # copy() 复制一份测试集，避免直接修改传入的数据。
    test_data = test_data.copy()

    # Name 是乘客姓名，第一版暂时不提取称谓，先删除。
    # Ticket 是船票号，格式比较杂，第一版也先删除。
    drop_columns = ["Name", "Ticket"]

    # drop(columns=...) 删除训练集中的指定列。
    train_data = train_data.drop(columns=drop_columns)

    # 测试集也删除同样的列，保证训练集和测试集字段一致。
    test_data = test_data.drop(columns=drop_columns)

    # map() 可以按照字典映射替换值。
    # male -> 0，female -> 1。
    train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})

    # 测试集 Sex 使用同样的映射规则。
    test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})

    # get_dummies() 用来做 one-hot 编码。
    # columns=["Embarked"] 表示只对 Embarked 这一列编码。
    train_data = pd.get_dummies(train_data, columns=["Embarked"])

    # 测试集也对 Embarked 做 one-hot 编码。
    test_data = pd.get_dummies(test_data, columns=["Embarked"])

    # align() 让训练集和测试集拥有完全一样的特征列。
    # join="left" 表示以训练集的列为准。
    # axis=1 表示按列对齐。
    # fill_value=0 表示测试集缺少的列用 0 补上。
    train_data, test_data = train_data.align(
        test_data,
        join="left",
        axis=1,
        fill_value=0,
    )

    # Survived 是标签列，也就是模型要学习预测的目标。
    y = train_data["Survived"]

    # drop(columns=["Survived"]) 从训练集中删除标签列，剩下的就是特征 X。
    X = train_data.drop(columns=["Survived"])

    # 测试集没有 Survived 标签。
    # 但 align 后可能出现 Survived 列，所以这里如果存在就删除。
    if "Survived" in test_data.columns:
        test_data = test_data.drop(columns=["Survived"])

    # 返回训练特征 X、训练标签 y、测试特征 test_data。
    return X, y, test_data


def show_result(X: pd.DataFrame, y: pd.Series, test_data: pd.DataFrame) -> None:
    """打印编码后的数据情况。"""
    # 打印训练特征形状。
    print("训练特征 X 形状：", X.shape)

    # 打印训练标签形状。
    print("训练标签 y 形状：", y.shape)

    # 打印测试特征形状。
    print("测试特征形状：", test_data.shape)

    # 查看编码后的训练特征列名。
    print("\n编码后的特征列：")
    print(X.columns.tolist())

    # 查看训练特征前 5 行。
    print("\n训练特征前 5 行：")
    print(X.head())

    # 查看训练标签前 5 行。
    print("\n训练标签前 5 行：")
    print(y.head())

    # 确认编码后是否还存在缺失值。
    print("\n训练特征缺失值总数：", X.isnull().sum().sum())
    print("测试特征缺失值总数：", test_data.isnull().sum().sum())


def main() -> None:
    """脚本入口函数。"""
    # 加载第二步脚本，这样可以复用里面的 load_data() 和 preprocess_data()。
    prepare_features = load_prepare_features_module()

    # 先读取原始训练集和测试集。
    train_data, test_data = prepare_features.load_data()

    # 复用第二步的基础预处理：删除 PassengerId/Cabin，填充缺失值。
    train_data, test_data, _ = prepare_features.preprocess_data(train_data, test_data)

    # 对预处理后的数据进行类别编码，并分离 X 和 y。
    X, y, test_data = encode_features(train_data, test_data)

    # 打印编码结果，确认字段和缺失值情况。
    show_result(X, y, test_data)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
