"""使用多种指标做 Titanic 决策树交叉验证。

本脚本接在 08_cross_validation.py 后面使用。

当前目标：
    1. 对不同 max_depth 的决策树做 5 折交叉验证。
    2. 同时观察 accuracy、precision、recall、f1。
    3. 理解不同指标下，最优超参数可能不同。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来整理多指标交叉验证结果。
import pandas as pd

# DecisionTreeClassifier 是 sklearn 中的决策树分类模型。
from sklearn.tree import DecisionTreeClassifier

# StratifiedKFold 用来做分层 K 折交叉验证。
from sklearn.model_selection import StratifiedKFold

# cross_validate 可以一次性计算多个评分指标。
from sklearn.model_selection import cross_validate


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


def load_all_training_data():
    """加载完整训练数据 X 和 y。"""
    # 加载第五步训练脚本模块。
    train_tree = load_train_tree_module()

    # 复用第五步的数据加载和编码函数。
    X, y, _ = train_tree.load_encoded_data()

    # 返回完整训练特征和训练标签。
    return X, y


def evaluate_depth_with_metrics(X, y, max_depth) -> dict:
    """对指定 max_depth 做多指标 5 折交叉验证。"""
    # 创建决策树模型。
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=22,
    )

    # StratifiedKFold 保证每一折里 0/1 标签比例尽量接近整体比例。
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=22,
    )

    # scoring 定义要同时计算的指标。
    # accuracy：整体准确率。
    # precision：预测为生还的人里，有多少是真的生还。
    # recall：真实生还的人里，有多少被模型找出来。
    # f1：precision 和 recall 的综合平衡。
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # cross_validate() 可以一次性返回多个指标的每折分数。
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
    )

    # cross_validate() 返回的 key 形如 test_accuracy、test_precision。
    return {
        "max_depth": max_depth,
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
    }


def compare_depths_with_metrics() -> pd.DataFrame:
    """比较不同 max_depth 在多个指标上的表现。"""
    # 加载完整训练数据。
    X, y = load_all_training_data()

    # 准备要尝试的树深度。
    depths = [1, 2, 3, 4, 5, 6, 7, None]

    # results 用来保存每个深度的多指标结果。
    results = []

    # 逐个深度做多指标交叉验证。
    for depth in depths:
        # 计算当前深度下的多个指标。
        result = evaluate_depth_with_metrics(X, y, depth)

        # 保存当前结果。
        results.append(result)

    # 转成 DataFrame，方便查看。
    return pd.DataFrame(results)


def show_best_by_metric(result_table: pd.DataFrame) -> None:
    """分别打印每个指标下表现最好的 max_depth。"""
    # 依次查看这四个指标。
    for metric in ["accuracy", "precision", "recall", "f1"]:
        # idxmax() 找到当前指标最大值所在行。
        best_index = result_table[metric].idxmax()

        # loc[] 取出这一行。
        best_row = result_table.loc[best_index]

        # 打印当前指标下的最佳深度和对应分数。
        print(f"\n按 {metric} 选择的最佳配置：")
        print(best_row)


def main() -> None:
    """脚本入口函数。"""
    # 设置 pandas 显示所有列。
    pd.set_option("display.max_columns", None)

    # 设置显示宽度，避免表格换行太乱。
    pd.set_option("display.width", 120)

    # 比较多个深度在多个指标上的表现。
    result_table = compare_depths_with_metrics()

    # 打印完整结果表。
    print("不同 max_depth 的多指标 5 折交叉验证结果：")
    print(result_table)

    # 分别打印每个指标下的最佳配置。
    show_best_by_metric(result_table)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
