# ml_study

## 当前学习位置

```text
机器学习概述
-> KNN 分类与回归
-> 特征缩放、Iris 训练 / 预测 / 评估 / 可视化
-> 交叉验证与网格搜索
-> 线性回归
-> 逻辑回归（当前）
```

当前逻辑回归已经完成的内容：

- 数学基础：概率、条件概率、伯努利分布、odds / log-odds、sigmoid、决策边界
- 交叉熵与损失函数
- 梯度与参数更新
- NumPy 手写逻辑回归 notebook
- `logistic_regression_numpy_manual.py` 手搓脚本正在继续完善

## 数学主线

线性回归阶段主要对应 Gilbert Strang《Introduction to Linear Algebra》第 4 章：

```text
正交
-> 投影
-> 最小二乘
-> Gram-Schmidt / QR
```

逻辑回归阶段的主线则转向：

```text
线性分数 z = Xw + b
-> sigmoid
-> 概率输出
-> 交叉熵
-> 梯度下降
```

## 目录说明

- `basic_knowledge/`
  - 机器学习概述
- `KNN_algorithm/`
  - KNN、标准化、交叉验证、网格搜索、Iris 实战
- `linear_regressor/`
  - 线性回归完整学习过程
- `logistic_regressor/`
  - 逻辑回归学习笔记与手写实现

## 当前重点文件

- [第一章 机器学习概述](basic_knowledge/第一章%20机器学习概述.md)
- [KNN算法](KNN_algorithm/KNN算法.md)
- [交叉验证与网格搜索学习笔记](KNN_algorithm/交叉验证与网格搜索学习笔记.md)

- [线性回归与线性代数学习地图](linear_regressor/线性回归与线性代数学习地图.md)
- [线性回归第一课](linear_regressor/线性回归第一课.md)
- [线性回归第二课_损失函数](linear_regressor/线性回归第二课_损失函数.md)
- [线性回归第三课_正规方程](linear_regressor/线性回归第三课_正规方程.md)
- [线性回归第四课_多元线性回归](linear_regressor/线性回归第四课_多元线性回归.md)
- [线性回归第五课_梯度下降](linear_regressor/线性回归第五课_梯度下降.md)
- [线性回归第六课_正则化](linear_regressor/线性回归第六课_正则化.ipynb)

- [逻辑回归第一课_数学基础](logistic_regressor/逻辑回归第一课_数学基础.ipynb)
- [逻辑回归第二课_交叉熵与损失函数](logistic_regressor/逻辑回归第二课_交叉熵与损失函数.ipynb)
- [逻辑回归第三课_梯度与参数更新](logistic_regressor/逻辑回归第三课_梯度与参数更新.ipynb)
- [逻辑回归第四课_手写逻辑回归](logistic_regressor/逻辑回归第四课_手写逻辑回归.ipynb)
- [logistic_regression_numpy_manual.py](logistic_regressor/logistic_regression_numpy_manual.py)

## 最近进度

### 2026/7/10

- 已将逻辑回归拆分为多份独立 notebook，避免内容混杂
- 已修正逻辑回归公式渲染问题，统一为可正常显示的数学格式
- 已完成逻辑回归前四课
- 已开始用 `NumPy` 和 `sklearn` 结合手搓逻辑回归脚本
- 当前脚本已完成：
  - 数据集构造
  - 训练集 / 测试集划分
  - 标准化
  - 设计矩阵构造
  - `sigmoid`
  - 交叉熵损失函数
  - 梯度
  - 参数更新
  - 训练循环

## 下一步

当前最直接的下一步是继续完善：

- `logistic_regression_numpy_manual.py`

后续计划：

- 训练完成后的预测与准确率评估
- 决策边界可视化
- `sklearn` 版逻辑回归实战
- 分类评估指标：混淆矩阵、precision、recall、F1
