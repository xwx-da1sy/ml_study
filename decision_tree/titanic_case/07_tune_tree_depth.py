"""比较不同决策树深度的模型效果。

本脚本接在 05_train_decision_tree.py 后面使用。

当前目标：
    1. 使用同一份训练集/验证集切分。
    2. 训练不同 max_depth 的决策树。
    3. 对比训练集准确率和验证集准确率。

为什么要这样做：
    决策树越深，模型越复杂。
    如果训练集准确率持续升高，但验证集准确率不升反降，就说明可能开始过拟合。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来整理调参结果表格。
import pandas as pd

# DecisionTreeClassifier 是 sklearn 中的决策树分类模型。
from sklearn.tree import DecisionTreeClassifier

# accuracy_score 用来计算准确率。
from sklearn.metrics import accuracy_score


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# 第五步训练脚本路径。
TRAIN_TREE_PATH = BASE_DIR / "05_train_decision_tree.py"


def load_train_tree_module():
    """从 05_train_decision_tree.py 文件中加载函数。"""
    # spec_from_file_location() 根据文件路径创建模块说明对象。
    spec = importlib.util.spec_from_file_location(
        "train_decision_tree",
        TRAIN_TREE_PATH,
    )

    # module_from_spec() 根据模块说明对象创建模块对象。
    module = importlib.util.module_from_spec(spec)

    # exec_module() 执行模块文件，让里面的函数可以被调用。
    spec.loader.exec_module(module)

    # 返回加载好的模块。
    return module


def load_train_valid_data():
    """加载编码后的数据，并切分训练集和验证集。"""
    # 加载第五步训练脚本模块。
    train_tree = load_train_tree_module()

    # 复用第五步的数据加载和编码函数。
    X, y, _ = train_tree.load_encoded_data()

    # 复用第五步的训练集/验证集切分函数。
    X_train, X_valid, y_train, y_valid = train_tree.split_train_valid(X, y)

    # 返回切分好的训练集和验证集。
    return X_train, X_valid, y_train, y_valid


def train_and_evaluate_by_depth(
    X_train,
    X_valid,
    y_train,
    y_valid,
    max_depth,
) -> dict:
    """训练指定深度的决策树，并返回评估结果。"""
    # 创建决策树模型。
    # max_depth 控制树的最大深度。
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=22,
    )

    # fit() 使用训练集训练模型。
    model.fit(X_train, y_train)

    # 在训练集上预测，用来看模型对已见数据拟合得怎么样。
    train_pred = model.predict(X_train)

    # 在验证集上预测，用来看模型对未见数据泛化得怎么样。
    valid_pred = model.predict(X_valid)

    # 计算训练集准确率。
    train_accuracy = accuracy_score(y_train, train_pred)

    # 计算验证集准确率。
    valid_accuracy = accuracy_score(y_valid, valid_pred)

    # 返回一个字典，方便后面整理成 DataFrame。
    return {
        "max_depth": max_depth,
        "actual_depth": model.get_depth(),
        "leaf_count": model.get_n_leaves(),
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
    }


def tune_depths() -> pd.DataFrame:
    """比较多个 max_depth 的模型效果。"""
    # 加载训练集和验证集。
    X_train, X_valid, y_train, y_valid = load_train_valid_data()

    # 准备要尝试的树深度。
    # None 表示不限制最大深度。
    depths = [1, 2, 3, 4, 5, 6, 7, None]

    # results 用来保存每一种深度的评估结果。
    results = []

    # 逐个尝试不同的 max_depth。
    for depth in depths:
        # 训练并评估当前深度的模型。
        result = train_and_evaluate_by_depth(
            X_train,
            X_valid,
            y_train,
            y_valid,
            depth,
        )

        # 把当前结果加入列表。
        results.append(result)

    # 把结果列表转换成 DataFrame，方便打印和比较。
    return pd.DataFrame(results)


def main() -> None:
    """脚本入口函数。"""
    # 比较不同 max_depth 的效果。
    result_table = tune_depths()

    # 打印完整结果表。
    print("不同 max_depth 的效果对比：")
    print(result_table)

    # idxmax() 找到验证集准确率最高的那一行索引。
    best_index = result_table["valid_accuracy"].idxmax()

    # loc[] 根据索引取出验证集表现最好的一行。
    best_row = result_table.loc[best_index]

    # 打印最佳深度。
    print("\n验证集准确率最高的配置：")
    print(best_row)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
