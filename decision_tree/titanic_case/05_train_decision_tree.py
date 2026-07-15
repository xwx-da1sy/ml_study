"""训练 Titanic 决策树分类模型。

本脚本接在 03_encode_features.py 后面使用。

当前目标：
    1. 获取已经完成缺失值处理和类别编码的数据。
    2. 将训练数据划分成训练集和验证集。
    3. 使用 DecisionTreeClassifier 训练一个决策树模型。
    4. 使用验证集评估模型效果。

说明：
    决策树不强制要求标准化，所以这里直接使用 03_encode_features.py 的编码结果。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来处理表格数据。
import pandas as pd

# DecisionTreeClassifier 是 sklearn 中的决策树分类模型。
from sklearn.tree import DecisionTreeClassifier

# accuracy_score 用来计算准确率。
from sklearn.metrics import accuracy_score

# classification_report 用来输出 precision、recall、f1-score 等指标。
from sklearn.metrics import classification_report

# confusion_matrix 用来生成混淆矩阵。
from sklearn.metrics import confusion_matrix

# train_test_split 用来把训练数据拆成训练集和验证集。
from sklearn.model_selection import train_test_split


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

    # 第三步脚本会继续复用第二步脚本。
    prepare_features = encode_features.load_prepare_features_module()

    # 读取原始训练集和测试集。
    train_data, test_data = prepare_features.load_data()

    # 做基础预处理：删除 PassengerId/Cabin，填充缺失值。
    train_data, test_data, _ = prepare_features.preprocess_data(train_data, test_data)

    # 做类别编码，并分离训练特征 X 和训练标签 y。
    X, y, test_data = encode_features.encode_features(train_data, test_data)

    # 返回训练特征、训练标签、测试特征。
    return X, y, test_data


def split_train_valid(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """将训练数据拆分成训练集和验证集。"""
    # train_test_split() 用来切分数据。
    # test_size=0.2 表示拿 20% 数据作为验证集。
    # random_state=22 表示固定随机种子，保证每次切分结果一致。
    # stratify=y 表示按照 y 的类别比例分层抽样，让训练集和验证集类别比例接近。
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=22,
        stratify=y,
    )

    # 返回切分后的四份数据。
    return X_train, X_valid, y_train, y_valid


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """训练一个决策树分类模型。"""
    # 创建决策树分类器。
    # criterion="gini" 表示使用 CART 常用的 Gini 指数选择划分。
    # max_depth=6 表示树最多长 6 层，防止第一版模型过拟合太严重。
    # random_state=22 表示固定随机种子，保证结果可复现。
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        random_state=22,
    )

    # fit() 用训练特征 X_train 和训练标签 y_train 训练模型。
    model.fit(X_train, y_train)

    # 返回训练好的模型。
    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> None:
    """在验证集上评估决策树模型。"""
    # predict() 使用训练好的模型对验证集进行预测。
    y_pred = model.predict(X_valid)

    # accuracy_score() 计算预测正确的比例。
    accuracy = accuracy_score(y_valid, y_pred)

    # 打印准确率。
    print("验证集准确率：", accuracy)

    # classification_report() 输出更详细的分类指标。
    print("\n分类报告：")
    print(classification_report(y_valid, y_pred))

    # confusion_matrix() 输出混淆矩阵。
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_pred))


def show_tree_info(model: DecisionTreeClassifier, feature_names: list[str]) -> None:
    """打印决策树模型的基础信息。"""
    # get_depth() 查看树的实际深度。
    print("\n决策树实际深度：", model.get_depth())

    # get_n_leaves() 查看叶子节点数量。
    print("叶子节点数量：", model.get_n_leaves())

    # feature_importances_ 表示模型认为每个特征的重要程度。
    importances = pd.Series(model.feature_importances_, index=feature_names)

    # sort_values(ascending=False) 按重要性从大到小排序。
    importances = importances.sort_values(ascending=False)

    # 打印特征重要性。
    print("\n特征重要性：")
    print(importances)


def main() -> None:
    """脚本入口函数。"""
    # 加载已经完成缺失值处理和类别编码的数据。
    X, y, _ = load_encoded_data()

    # 将训练数据切分成训练集和验证集。
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    # 训练决策树模型。
    model = train_decision_tree(X_train, y_train)

    # 在验证集上评估模型效果。
    evaluate_model(model, X_valid, y_valid)

    # 打印树深度、叶子节点数量和特征重要性。
    show_tree_info(model, X.columns.tolist())


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
