"""
KNN 交叉验证与网格搜索

本文件对应黑马程序员机器学习课程中 KNN 后半部分：
    1. 交叉验证
    2. 网格搜索
    3. 使用 Pipeline 防止数据泄露

对应 Markdown：
    - 交叉验证与网格搜索学习笔记.md：第 2-3 节 交叉验证
    - 交叉验证与网格搜索学习笔记.md：第 4 节 网格搜索
    - 交叉验证与网格搜索学习笔记.md：第 5 节 为什么要用 Pipeline

核心目标：
    不再手动猜 n_neighbors，而是让 GridSearchCV 在多个 K 值中选择验证效果最好的参数。

线性代数视角：
    X 是特征矩阵，形状是 (样本数, 特征数)。
    交叉验证每次切换参与训练和验证的“行”。
    标准化处理的是每一列特征，也就是改变特征空间的坐标尺度。
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. 加载鸢尾花数据集，对应笔记第 8 节：X 是 (150, 4) 特征矩阵，y 是标签向量。
iris = load_iris()
X = iris.data
y = iris.target

print("原始特征矩阵 X 的形状：", X.shape)
print("标签向量 y 的形状：", y.shape)

# 2. 划分训练集和测试集，对应笔记第 7 节：测试集只用于最终评估。
# stratify=y 表示按照 y 的类别比例分层抽样，尽量保持训练集和测试集中类别比例一致。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=22,
    stratify=y,
)

print("训练集特征矩阵 X_train 的形状：", X_train.shape)
print("测试集特征矩阵 X_test 的形状：", X_test.shape)

# 3. 创建 Pipeline，对应笔记第 5 节：把标准化和模型训练绑定成一个整体。
# Pipeline 会把“标准化”和“KNN 模型”串成一个整体。
# 在交叉验证的每一折中，它都会只用当前训练部分 fit 标准化器，
# 再用同一个标准化规则 transform 当前验证部分，从而避免数据泄露。
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)

# 4. 设置要搜索的参数网格，对应笔记第 4 节：让模型尝试多个候选 K 值。
# 参数名 knn__n_neighbors 的含义：
#     knn             -> Pipeline 中 KNN 这一步的名字
#     n_neighbors     -> KNeighborsClassifier 的 K 值参数
#     knn__n_neighbors -> “knn 这一步里的 n_neighbors 参数”
param_grid = {
    "knn__n_neighbors": range(1, 21),
}

# 5. 创建网格搜索对象，对应笔记第 3 节：cv=5 表示五折交叉验证。
# cv=5 表示五折交叉验证。
# scoring="accuracy" 表示用准确率评价每个 K 值。
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)

# 6. 在训练集上搜索最佳 K 值：每个 K 都会做 5 次训练/验证，再比较平均验证分数。
grid_search.fit(X_train, y_train)

# 7. 输出交叉验证结果
print("最佳参数：", grid_search.best_params_)
print(f"交叉验证最佳平均准确率：{grid_search.best_score_:.2%}")

# 8. 使用最佳模型在测试集上做最终评估，对应笔记第 7 节：测试集是最后一次考试。
test_score = grid_search.score(X_test, y_test)
print(f"最终测试集准确率：{test_score:.2%}")

# 9. 展示每个 K 值对应的平均交叉验证准确率
print("\n不同 K 值的交叉验证平均准确率：")
for params, mean_score in zip(
    grid_search.cv_results_["params"],
    grid_search.cv_results_["mean_test_score"],
):
    k = params["knn__n_neighbors"]
    print(f"K = {k:2d}, mean_accuracy = {mean_score:.2%}")
