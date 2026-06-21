"""
用KNN算法预测鸢尾花

主要有一下几个步骤：

1. 导包

这里导包包含几个工具：

sklearn.datasets：数据集工具包，包含了很多经典的数据集，可以直接加载使用

sklearn.neighbors: KNN算法的模型，估计器

用来区分测试集和训练集的工具

用来进行数据标准化的工具

2. 加载数据集

3. 提取特征和标签

4. 分割训练集和测试集

5. 对数据进行标准化

注意这里进行测试集标准化的时候标准差和平均数都使用的是训练集中的数据，这样可以有效防止数据泄露

6. 创建KNN估计器

7. 训练模型（fit）

8. 预测与评估
"""


# 1. 导包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 2. 加载数据集
iris = load_iris()

print(type(iris))
# 展示iris这个Bunch类的对象中存储着什么信息
print(iris.keys())
print(iris.data.shape)
print(iris.target.shape)

# 3. 提取特征和标签
x = iris.data
y = iris.target

print(x[0])
print(y[0])
print(iris.target_names[y[0]])

# 4. 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    # 注意：这里random_state的值可以随便设置，设置成什么都行，主要是为了保证每次运行代码的时候分割的结果都是一样的
    # random_state表示的是随机种子
    random_state=22,
)

print("-----------------------------------------------------")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 5. 对数据进行标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("-----------------------------------------------------")
print(x_train_scaled.shape)
print(x_test_scaled.shape)
# 输出标准化之后每一列的平均值
print(x_train_scaled.mean(axis=0))
# 输出标准化之后每一列的标准差
print(x_train_scaled.std(axis=0))

# 6. 创建KNN估计器
# 注意鸢尾花数据集是一个多分类问题，所以我们使用KNeighborsClassifier这个分类器来进行预测
knn = KNeighborsClassifier(n_neighbors=5)

print("-----------------------------------------------------")
print(knn)
print(knn.n_neighbors)

# 7. 使用训练集训练模型
knn.fit(x_train_scaled, y_train)

# 8. 使用测试集预测并评估模型
y_pred = knn.predict(x_test_scaled)
accuracy = knn.score(x_test_scaled, y_test)

print("-----------------------------------------------------")
print("预测标签：", y_pred)
print("真实标签：", y_test)
print(f"测试集准确率：{accuracy:.2%}")
