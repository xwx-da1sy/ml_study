"""训练最终 Titanic 决策树模型并生成预测文件。

本脚本是 Titanic 决策树案例的收尾步骤。

当前目标：
    1. 使用完整训练集训练最终模型。
    2. 使用交叉验证选出的 max_depth=6。
    3. 对 titanic_test.csv 进行预测。
    4. 生成 submission.csv，格式为 PassengerId + Survived。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# joblib 用来保存训练好的 sklearn 模型。
import joblib

# pandas 用来创建和保存预测结果表格。
import pandas as pd

# DecisionTreeClassifier 是 sklearn 中的决策树分类模型。
from sklearn.tree import DecisionTreeClassifier


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# 第三步编码脚本路径。
ENCODE_FEATURES_PATH = BASE_DIR / "03_encode_features.py"

# 输出目录，用来保存预测文件和模型文件。
OUTPUT_DIR = BASE_DIR / "output"

# 最终预测文件保存路径。
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# 最终模型保存路径。
MODEL_PATH = OUTPUT_DIR / "titanic_decision_tree_model.joblib"


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


def load_final_training_data():
    """加载最终训练模型需要的 X、y、测试集特征和测试集 PassengerId。"""
    # 加载第三步编码脚本模块。
    encode_features = load_encode_features_module()

    # 第三步脚本里可以继续加载第二步预处理脚本。
    prepare_features = encode_features.load_prepare_features_module()

    # 读取原始训练集和测试集。
    train_data, test_data = prepare_features.load_data()

    # 对原始数据做基础预处理。
    # test_passenger_id 是测试集乘客编号，最后生成 submission.csv 要用。
    train_data, test_data, test_passenger_id = prepare_features.preprocess_data(
        train_data,
        test_data,
    )

    # 做类别编码，并分离训练特征 X 和标签 y。
    X, y, test_data = encode_features.encode_features(train_data, test_data)

    # 返回最终训练和预测需要的所有数据。
    return X, y, test_data, test_passenger_id


def train_final_model(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    """使用完整训练集训练最终决策树模型。"""
    # 创建最终决策树模型。
    # max_depth=6 来自前面 5 折交叉验证中表现较好的配置。
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        random_state=22,
    )

    # fit() 使用完整训练集训练最终模型。
    model.fit(X, y)

    # 返回训练好的模型。
    return model


def predict_test_data(
    model: DecisionTreeClassifier,
    test_data: pd.DataFrame,
) -> pd.Series:
    """使用最终模型预测测试集。"""
    # predict() 对测试集每一行预测 Survived。
    predictions = model.predict(test_data)

    # 把 numpy 数组转成 pandas Series，方便后续组成表格。
    return pd.Series(predictions, name="Survived")


def save_submission(
    test_passenger_id: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """保存 Kaggle 风格的预测结果文件。"""
    # mkdir() 创建输出目录。
    # exist_ok=True 表示目录已存在时不报错。
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 创建提交结果表。
    # 第一列 PassengerId，第二列 Survived。
    submission = pd.DataFrame(
        {
            "PassengerId": test_passenger_id,
            "Survived": predictions,
        }
    )

    # to_csv() 把结果保存成 CSV。
    # index=False 表示不要额外保存 DataFrame 的行索引。
    submission.to_csv(SUBMISSION_PATH, index=False)

    # 返回 submission，方便打印查看。
    return submission


def save_model(model: DecisionTreeClassifier) -> None:
    """保存训练好的最终模型。"""
    # mkdir() 确保输出目录存在。
    OUTPUT_DIR.mkdir(exist_ok=True)

    # joblib.dump() 把 sklearn 模型保存到文件。
    joblib.dump(model, MODEL_PATH)


def main() -> None:
    """脚本入口函数。"""
    # 加载最终训练和预测需要的数据。
    X, y, test_data, test_passenger_id = load_final_training_data()

    # 使用完整训练集训练最终模型。
    model = train_final_model(X, y)

    # 对测试集进行预测。
    predictions = predict_test_data(model, test_data)

    # 保存 submission.csv。
    submission = save_submission(test_passenger_id, predictions)

    # 保存训练好的模型文件。
    save_model(model)

    # 打印最终模型基本信息。
    print("最终模型训练完成。")
    print("决策树深度：", model.get_depth())
    print("叶子节点数量：", model.get_n_leaves())

    # 打印预测文件路径。
    print("\n预测文件已保存：", SUBMISSION_PATH)

    # 打印模型文件路径。
    print("模型文件已保存：", MODEL_PATH)

    # 查看预测结果前 5 行。
    print("\n预测结果前 5 行：")
    print(submission.head())

    # 查看预测类别分布。
    print("\n预测类别分布：")
    print(submission["Survived"].value_counts())


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
