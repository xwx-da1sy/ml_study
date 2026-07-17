# ml_study

## 当前学习位置

```text
机器学习概述
-> KNN 分类与回归
-> 特征缩放、Iris 训练 / 预测 / 评估 / 可视化
-> 交叉验证与网格搜索
-> 线性回归
-> 逻辑回归
-> 决策树
-> 集成学习（当前）
```

当前集成学习已经开始的内容：

- 已阅读机器学习概述、决策树与随机森林相关背景
- 已新建 `ensemble_learning/` 章节目录
- 已建立 Bagging、Boosting、Stacking 三条学习主线
- 已完成集成学习第一课：基本概念
- 已开始第二课：Bagging
- 已开始第三课：随机森林复盘
- 已开始第四课：AdaBoost

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
- `decision_tree/`
  - 决策树、ID3、C4.5、CART、随机森林与 Titanic 案例
- `ensemble_learning/`
  - 集成学习路线、Bagging、Boosting、Stacking 相关笔记

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
- [集成学习学习路线](ensemble_learning/集成学习学习路线.ipynb)
- [集成学习第一课_基本概念](ensemble_learning/集成学习第一课_基本概念.ipynb)
- [集成学习第二课_Bagging](ensemble_learning/集成学习第二课_Bagging.ipynb)
- [集成学习第三课_随机森林复盘](ensemble_learning/集成学习第三课_随机森林复盘.ipynb)
- [集成学习第四课_AdaBoost](ensemble_learning/集成学习第四课_AdaBoost.ipynb)

## 最近进度

### 2026/7/17

- 已从决策树与随机森林过渡到集成学习章节
- 已新建 `ensemble_learning/` 目录
- 已建立集成学习路线：Bagging、Boosting、Stacking
- 已完成第一课：集成学习基本概念
- 已开始第二课：Bagging，并加入 Bootstrap、投票、平均、OOB 与 sklearn 实战示例
- 已开始第三课：随机森林复盘，并加入 Bagging / 随机森林对比、OOB 与特征重要性示例
- 已开始第四课：AdaBoost，并加入样本权重、弱学习器权重、加权投票、数学推导、手算例子与 sklearn 实战示例

### 2026/7/10

- 已将逻辑回归拆分为多份独立 notebook，避免内容混杂
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

当前最直接的下一步是继续学习：

- 运行并理解 `集成学习第二课_Bagging.ipynb`
- Bagging 与随机森林复盘
- 运行并理解 `集成学习第三课_随机森林复盘.ipynb`
- 运行并理解 `集成学习第四课_AdaBoost.ipynb`
- GBDT：从残差理解梯度提升树
- sklearn 集成学习实战
