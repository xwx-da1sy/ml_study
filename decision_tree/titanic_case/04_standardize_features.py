"""Titanic 特征标准化。

本脚本接在 03_encode_features.py 后面使用。

说明：
    决策树模型通常不强制要求标准化，因为决策树是按照特征阈值切分数据。
    但是标准化是机器学习中非常常见的数据处理步骤。
    如果后面要对比 KNN、逻辑回归、SVM 等模型，标准化就很重要。

当前目标：
    1. 复用第三步得到的 X、y 和 test_data。
    2. 使用 StandardScaler 对特征做标准化。
    3. 打印标准化前后的数据，帮助理解标准化效果。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来保存和展示表格数据。
import pandas as pd

# StandardScaler 是 sklearn 中常用的标准化工具。
from sklearn.preprocessing import StandardScaler


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# 第三步编码脚本的文件路径。
ENCODE_FEATURES_PATH = BASE_DIR / "03_encode_features.py"


def load_encode_features_module():
    """从 03_encode_features.py 文件中加载函数。"""
    # spec_from_file_location() 根据文件路径创建模块说明对象。
    spec = importlib.util.spec_from_file_location(
        "encode_features",
        ENCODE_FEATURES_PATH,
    )

    # module_from_spec() 根据模块说明对象创建模块对象。
    module = importlib.util.module_from_spec(spec)

    # exec_module() 执行模块文件，让里面的函数可以被调用。
    spec.loader.exec_module(module)

    # 返回加载好的模块。
    return module


def load_encoded_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """加载已经完成类别编码的数据。"""
    # 加载第三步脚本模块。
    encode_features = load_encode_features_module()

    # 第三步脚本里又会加载第二步脚本模块。
    prepare_features = encode_features.load_prepare_features_module()

    # 读取原始 Titanic 训练集和测试集。
    train_data, test_data = prepare_features.load_data()

    # 做第二步基础预处理：删列、填充缺失值。
    train_data, test_data, _ = prepare_features.preprocess_data(train_data, test_data)

    # 做第三步类别编码，并分离训练特征 X 和标签 y。
    X, y, test_data = encode_features.encode_features(train_data, test_data)

    # 返回编码后的训练特征、训练标签、测试特征。
    return X, y, test_data


def standardize_features(
    X: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """对训练特征和测试特征做标准化。

    参数:
        X: 编码后的训练特征。
        test_data: 编码后的测试特征。

    返回值:
        X_scaled: 标准化后的训练特征。
        test_scaled: 标准化后的测试特征。
        scaler: 已经 fit 好的 StandardScaler 对象。
    """
    # 创建 StandardScaler 标准化器。
    scaler = StandardScaler()

    # fit_transform() 做两件事：
    # 1. fit：在训练集上计算每一列的平均值和标准差。
    # 2. transform：用这些平均值和标准差转换训练集。
    X_scaled_array = scaler.fit_transform(X)

    # transform() 只做转换，不重新计算平均值和标准差。
    # 测试集必须使用训练集算出来的标准化规则，避免数据泄漏。
    test_scaled_array = scaler.transform(test_data)

    # StandardScaler 返回的是 numpy 数组。
    # 这里把它转回 DataFrame，方便保留原来的列名。
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

    # 测试集也转回 DataFrame，列名和训练集保持一致。
    test_scaled = pd.DataFrame(test_scaled_array, columns=test_data.columns)

    # 返回标准化后的训练集、测试集，以及标准化器本身。
    return X_scaled, test_scaled, scaler


def show_result(
    X: pd.DataFrame,
    X_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    scaler: StandardScaler,
) -> None:
    """打印标准化前后的结果。"""
    # 查看标准化前的训练特征前 5 行。
    print("标准化前训练特征前 5 行：")
    print(X.head())

    # 查看标准化后的训练特征前 5 行。
    print("\n标准化后训练特征前 5 行：")
    print(X_scaled.head())

    # mean_ 是 StandardScaler 在训练集上学到的每列均值。
    print("\n训练集每个特征的均值：")
    print(pd.Series(scaler.mean_, index=X.columns))

    # scale_ 是 StandardScaler 在训练集上学到的每列标准差。
    print("\n训练集每个特征的标准差：")
    print(pd.Series(scaler.scale_, index=X.columns))

    # 标准化后，训练特征每列均值会非常接近 0。
    print("\n标准化后训练特征均值：")
    print(X_scaled.mean())

    # 标准化后，训练特征每列标准差会接近 1。
    print("\n标准化后训练特征标准差：")
    print(X_scaled.std())

    # 查看测试集标准化后的形状，确认列数没有变化。
    print("\n标准化后测试特征形状：", test_scaled.shape)


def main() -> None:
    """脚本入口函数。"""
    # 加载第三步处理好的编码数据。
    X, y, test_data = load_encoded_data()

    # 对训练特征和测试特征做标准化。
    X_scaled, test_scaled, scaler = standardize_features(X, test_data)

    # 打印标准化结果。
    show_result(X, X_scaled, test_scaled, scaler)

    # y 是标签，不需要标准化；这里打印一下形状确认它还在。
    print("\n训练标签 y 形状：", y.shape)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
