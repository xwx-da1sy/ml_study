"""可视化 Titanic 决策树模型。

本脚本接在 05_train_decision_tree.py 后面使用。

当前目标：
    1. 复用第五步的训练流程，得到训练好的决策树模型。
    2. 使用 sklearn.tree.plot_tree() 画出决策树。
    3. 将图片保存到 titanic_case/output 文件夹中。
"""

# importlib.util 用来从指定文件路径加载 Python 脚本。
import importlib.util

# pathlib.Path 用来处理文件路径。
from pathlib import Path

# matplotlib.pyplot 用来创建和保存图片。
import matplotlib.pyplot as plt

# plot_tree 是 sklearn 提供的决策树可视化函数。
from sklearn.tree import plot_tree


# 当前脚本所在目录：decision_tree/titanic_case。
BASE_DIR = Path(__file__).resolve().parent

# 第五步训练脚本的文件路径。
TRAIN_TREE_PATH = BASE_DIR / "05_train_decision_tree.py"

# 图片输出目录。
OUTPUT_DIR = BASE_DIR / "output"

# 决策树图片保存路径。
TREE_IMAGE_PATH = OUTPUT_DIR / "titanic_decision_tree.png"


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


def train_model_for_plot():
    """训练一个决策树模型，并返回画图需要的数据。"""
    # 加载第五步训练脚本。
    train_tree = load_train_tree_module()

    # 复用第五步的数据加载和编码逻辑。
    X, y, _ = train_tree.load_encoded_data()

    # 复用第五步的训练集/验证集划分函数。
    X_train, X_valid, y_train, y_valid = train_tree.split_train_valid(X, y)

    # 复用第五步的模型训练函数。
    model = train_tree.train_decision_tree(X_train, y_train)

    # 返回模型、特征名、验证集和验证标签。
    # 验证集后续如果想继续评估或对照，可以直接使用。
    return model, X.columns.tolist(), X_valid, y_valid


def save_tree_image(model, feature_names: list[str]) -> Path:
    """画出决策树并保存为 PNG 图片。"""
    # mkdir() 创建输出目录。
    # exist_ok=True 表示目录已存在时不报错。
    OUTPUT_DIR.mkdir(exist_ok=True)

    # figure() 创建画布。
    # figsize 控制图片大小，树节点多时需要大一点。
    plt.figure(figsize=(22, 12))

    # plot_tree() 把 sklearn 的决策树模型画出来。
    # feature_names 显示特征名。
    # class_names 显示类别名。
    # 这里使用英文，是为了避免 matplotlib 默认字体缺少中文导致图片出现方框。
    # filled=True 表示节点用颜色区分类别纯度。
    # rounded=True 表示节点边框使用圆角。
    # fontsize 控制字体大小。
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=10,
    )

    # tight_layout() 自动调整边距，尽量避免内容被裁剪。
    plt.tight_layout()

    # savefig() 将图片保存到指定路径。
    # dpi 控制图片清晰度。
    plt.savefig(TREE_IMAGE_PATH, dpi=200)

    # close() 关闭当前画布，释放内存。
    plt.close()

    # 返回图片路径，方便打印给用户。
    return TREE_IMAGE_PATH


def main() -> None:
    """脚本入口函数。"""
    # 训练模型，并拿到画图需要的特征名。
    model, feature_names, _, _ = train_model_for_plot()

    # 画出决策树并保存图片。
    image_path = save_tree_image(model, feature_names)

    # 打印保存位置。
    print("决策树图片已保存：", image_path)


# 只有直接运行本文件时才执行 main()。
if __name__ == "__main__":
    main()
