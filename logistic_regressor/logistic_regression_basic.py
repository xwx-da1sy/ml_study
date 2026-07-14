import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ========== 1. 导包 ==========
# numpy：做数组与后续矩阵运算
# load_iris：使用 sklearn 自带的 Iris 数据集


# ========== 2. 准备数据集 ==========
# Iris 原本有 3 个类别。
# 这里先只取前两个类别，变成一个标准的二分类问题，更适合逻辑回归入门。
iris = load_iris()

X = iris.data
y = iris.target

# 生成布尔掩码：y<2的位置为True（类别0和1），y>=2的位置为False（类别2，即virginica）
binary_mask = y < 2
# 仅保留类别0（setosa）和类别1（versicolor）的样本，剔除类别2（virginica），将三分类转为二分类
X = X[binary_mask]
y = y[binary_mask]


# ========== 3. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ========== 4. 标准化 ==========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========== 5. 创建 estimator ==========
estimator = LogisticRegression(random_state=42)

print("estimator:", estimator)


# ========== 6. 训练模型 ==========
estimator.fit(X_train, y_train)


# ========== 7. 预测 ==========
y_pred = estimator.predict(X_test)
y_prob = estimator.predict_proba(X_test)

print("y_test:", y_test)
print("y_pred:", y_pred)
print("first 5 predicted probabilities:")
print(y_prob[:5])


# ========== 8. 混淆矩阵 ==========
cm = confusion_matrix(y_test, y_pred)

print("confusion matrix:")
print(cm)


# ========== 9. 计算 TN / FP / FN / TP ==========
tn, fp, fn, tp = cm.ravel()

print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# ========== 10. 计算 accuracy / precision / recall / F1 ==========
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("accuracy =", accuracy)
print("precision =", precision)
print("recall =", recall)
print("f1 =", f1)


# ========== 11. ROC 曲线与 AUC ==========
y_score = y_prob[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)

print("fpr =", fpr)
print("tpr =", tpr)
print("thresholds =", thresholds)
print("auc =", auc)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, marker="o", label=f"ROC curve (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("logistic_regressor/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print("ROC curve saved to: logistic_regressor/roc_curve.png")
