"""查看 Titanic 数据集的基本情况。

本脚本只做“加载和观察数据”，不做缺失值处理、特征编码和模型训练。
这一版特意写了比较密的逐行注释，方便初学者理解每一步。
"""

# pathlib.Path 用来处理文件路径，比直接拼接字符串更稳定。
from pathlib import Path

# Optional 用来表示“这个值可能是某种类型，也可能是 None”。
from typing import Optional

# pandas 是数据分析里最常用的库，这里用它读取 CSV 和查看表格数据。
import pandas as pd


# __file__ 表示当前 Python 文件路径。
# resolve() 会把路径转换成绝对路径。
# parent 表示当前文件所在的文件夹。
BASE_DIR = Path(__file__).resolve().parent

# / 是 pathlib 提供的路径拼接写法。
# 这里表示 data 文件夹在当前脚本所在目录下面。
DATA_DIR = BASE_DIR / "data"

# TRAIN_CANDIDATES 保存训练集可能出现的路径。
# 这样写的好处是：文件名叫 titanic_train.csv 或 train.csv 都能识别。
TRAIN_CANDIDATES = [
    DATA_DIR / "titanic_train.csv",
    DATA_DIR / "train.csv",
    BASE_DIR / "titanic_train.csv",
    BASE_DIR / "train.csv",
]

# TEST_CANDIDATES 保存测试集可能出现的路径。
# 测试集找不到也没关系，本脚本可以只查看训练集。
TEST_CANDIDATES = [
    DATA_DIR / "titanic_test.csv",
    DATA_DIR / "test.csv",
    BASE_DIR / "titanic_test.csv",
    BASE_DIR / "test.csv",
]


def find_existing_path(candidates: list[Path]) -> Optional[Path]:
    """从候选路径中找到第一个真实存在的文件。"""
    # for 循环会依次取出 candidates 列表里的每一个路径。
    for path in candidates:
        # exists() 用来判断这个路径对应的文件是否真实存在。
        if path.exists():
            # 找到第一个存在的文件后，立即返回这个路径。
            return path

    # 如果循环结束还没有找到文件，就返回 None。
    return None


def load_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    """读取一个 CSV 文件，并打印它的路径和形状。"""
    # pd.read_csv() 用来读取 CSV 文件，返回一个 DataFrame 表格对象。
    data = pd.read_csv(path)

    # f-string 可以把变量直接放进字符串里，方便打印路径。
    print(f"{dataset_name}路径：{path}")

    # shape 是 DataFrame 的属性，结果是 (行数, 列数)。
    print(f"{dataset_name}形状：{data.shape}")

    # 把读取到的表格数据返回给调用者。
    return data


def show_basic_info(
    train_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
) -> None:
    """打印训练集和测试集的基础信息。"""
    # head() 默认查看 DataFrame 的前 5 行。
    print("\n训练集前 5 行：")
    print(train_data.head())

    # info() 用来查看字段名、非空数量、数据类型等信息。
    print("\n训练集字段信息：")
    train_data.info()

    # isnull() 判断每个位置是不是缺失值。
    # sum() 会按列统计缺失值数量。
    print("\n训练集缺失值统计：")
    print(train_data.isnull().sum())

    # 如果 test_data 不是 None，说明测试集文件也读取成功了。
    if test_data is not None:
        # 查看测试集前 5 行。
        print("\n测试集前 5 行：")
        print(test_data.head())

        # 查看测试集字段信息。
        print("\n测试集字段信息：")
        test_data.info()

        # 查看测试集缺失值统计。
        print("\n测试集缺失值统计：")
        print(test_data.isnull().sum())


def main() -> None:
    """脚本入口函数。"""
    # 从训练集候选路径中找到实际存在的训练集文件。
    train_path = find_existing_path(TRAIN_CANDIDATES)

    # 从测试集候选路径中找到实际存在的测试集文件。
    test_path = find_existing_path(TEST_CANDIDATES)

    # 如果没有找到训练集，就主动报错，因为没有训练集就无法继续。
    if train_path is None:
        # join() 把多个路径字符串用换行符连接起来，方便提示用户。
        expected_paths = "\n".join(str(path) for path in TRAIN_CANDIDATES)

        # raise 用来主动抛出异常，告诉用户缺少必要文件。
        raise FileNotFoundError(
            "没有找到 Titanic 训练集。\n"
            "请把数据文件放到下面任意一个位置：\n"
            f"{expected_paths}"
        )

    # 读取训练集 CSV。
    train_data = load_csv(train_path, "训练集")

    # 如果找到了测试集路径，就读取测试集；否则 test_data 设为 None。
    test_data = load_csv(test_path, "测试集") if test_path else None

    # 如果没有测试集，给出提示；这不影响查看训练集。
    if test_path is None:
        print("\n没有找到测试集文件，当前只查看训练集。")

    # 打印训练集和测试集的基础信息。
    show_basic_info(train_data, test_data)


# 这句表示：只有直接运行本文件时，才执行 main()。
# 如果这个文件被其他脚本 import，引入时不会自动执行 main()。
if __name__ == "__main__":
    main()
