"""K-means 第一课：使用二维模拟数据完成第一次聚类。"""

import os
from pathlib import Path

# 避免 Windows + MKL 环境在小数据集上产生已知的 KMeans 警告。
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# make_blobs 也会返回人工标签，但无监督训练不使用标签。
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.75,
    random_state=22,
)

model = KMeans(n_clusters=3, random_state=22, n_init=10)
labels = model.fit_predict(X)
centers = model.cluster_centers_

print("X 的形状:", X.shape)
print("前 10 个簇编号:", labels[:10])
print("最终质心:\n", centers)
print("簇内平方和 inertia:", round(model.inertia_, 2))

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=35, alpha=0.8)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=220,
    edgecolors="white",
    linewidths=1.5,
    label="centroids",
)
plt.title("K-means clustering: K = 3")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.tight_layout()

output_path = Path(__file__).with_name("kmeans_basic_result.png")
plt.savefig(output_path, dpi=150)
print("可视化已保存到:", output_path)
