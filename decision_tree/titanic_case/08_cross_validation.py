"""使用交叉验证选择 Titanic 决策树超参数。

本脚本接在 07_tune_tree_depth.py 后面使用。

当前目标：
    1. 使用完整的训练数据 X 和 y。
    2. 对不同 max_depth 的决策树做 5 折交叉验证。
    3. 比较每个 max_depth 的平均准确率。

为什么要交叉验证：
    单次训练集/验证集切分可能有偶然性。
    交叉验证会把数据分成多份，轮流做验证，结果通常更稳定。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# pandas 用来整理交叉验证结果表格。
import pandas as pd

# DecisionTreeClassifier 是 sklearn 中的决策树分类模型。
from sklearn.tree import DecisionTreeClassifier

# StratifiedKFold 用来做分层 K 折交叉验证。
from sklearn.model_selection import StratifiedKFold

# cross_val_score 用来执行交叉验证并返回每一折的得分。
from sklearn.model_selection import cross_val_score


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


def evaluate_depth_with_cv(X, y, max_depth) -> dict:
    """对指定 max_depth 做 5 折交叉验证。"""
    # 创建决策树模型。
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=22,
    )

    # StratifiedKFold 是分层 K 折交叉验证。
    # n_splits=5 表示分成 5 折。
    # shuffle=True 表示划分前先打乱数据。
    # random_state=22 固定随机种子，让结果可复现。
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=22,
    )

    # cross_val_score() 会自动完成：
    # 1. 按 cv 规则切分数据。
    # 2. 每一折训练一个模型。
    # 3. 每一折在验证折上打分。
    # scoring="accuracy" 表示使用准确率作为评分。
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    # scores 是 5 个准确率，mean() 是平均准确率，std() 是波动程度。
    return {
        "max_depth": max_depth,
        "fold_scores": scores,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
    }


def compare_depths_with_cv() -> pd.DataFrame:
    """比较多个 max_depth 的交叉验证结果。"""
    # 加载完整训练数据。
    X, y = load_all_training_data()

    # 准备要尝试的 max_depth。
    depths = [1, 2, 3, 4, 5, 6, 7, None]

    # results 用来保存每个深度的交叉验证结果。
    results = []

    # 逐个深度做交叉验证。
    for depth in depths:
        # 计算当前深度的交叉验证结果。
        result = evaluate_depth_with_cv(X, y, depth)

        # 把结果加入列表。
        results.append(result)

    # 把结果整理成 DataFrame。
    result_table = pd.DataFrame(results)

    # 为了打印更清楚，把每折分数转换成保留 4 位小数的列表。
    result_table["fold_scores"] = result_table["fold_scores"].apply(
        lambda scores: [round(score, 4) for score in scores]
    )

    # 返回结果表。
    return result_table


def main() -> None:
    """脚本入口函数。"""
    # 设置 pandas 显示选项，避免结果表中间列被省略成省略号。
    pd.set_option("display.max_columns", None)

    # 设置每列显示宽度，方便看到 fold_scores 的完整列表。
    pd.set_option("display.width", 120)

    # 比较不同深度的交叉验证结果。
    result_table = compare_depths_with_cv()

    # 打印完整结果。
    print("不同 max_depth 的 5 折交叉验证结果：")
    print(result_table)

    # idxmax() 找到平均准确率最高的行索引。
    best_index = result_table["mean_accuracy"].idxmax()

    # loc[] 根据索引取出最佳结果。
    best_row = result_table.loc[best_index]

    # 打印最佳配置。
    print("\n平均准确率最高的配置：")
    print(best_row)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
