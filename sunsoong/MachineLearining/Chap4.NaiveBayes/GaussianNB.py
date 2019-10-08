# GaussianNB就是先验为高斯分布(正态分布)的朴素贝叶斯,即每个标签的数据都是服从简单的正态分布
# 调用sklearn包对鸢尾花数据进行分类

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 导入数据集
from sklearn import datasets

iris = datasets.load_iris()
# 切分数据集
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, random_state=12)

# 建模
clf = GaussianNB()  # 初始化高斯
clf.fit(xtrain, ytrain)  # 开始训练集

# 在测试集上执行预测,proba导出的是每个样本属于某类的概率
clf.predict(xtest)
clf.predict(xtest)

# 测试精准度
print(accuracy_score(ytest, clf.predict(xtest)))
