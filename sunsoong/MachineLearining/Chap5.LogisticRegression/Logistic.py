import pandas as pd
import numpy as np

dataSet = pd.read_table('testSet.txt', header=None)
dataSet.columns = ['x1', 'x2', 'labels']
print(dataSet.shape)


def sigmoid(inX):
    """
    定义sigmoid函数
    :param inX: 数值型数据
    :return: 经过sigmoid函数计算后的函数值
    """
    s = 1 / 1 + np.exp(-inX)
    return s


def regularize(xMat):
    """
    标准化函数(期望为0,方差为1)
    :param xMat: 特征矩阵
    :return: 标准化之后的特征矩阵
    """
    inMat = xMat.copy()
    inMeans = np.mean(inMat, axis=0)
    inVar = np.std(inMat, axis=0)
    inMat = (inMat - inMeans) / inVar
    return inMat


def BGD_LR(dataSet, alpha=0.001, maxCycles=500):
    """
    使用BGD求解逻辑回归
    :param dataSet: 数据集
    :param alpha: 步长
    :param maxCycles: 最大迭代次数
    :return:  weights:各特征权重值
    """
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = regularize(xMat)  # 标椎化
    m, n = xMat.shape
    weights = np.zeros((n, 1))
    for i in range(maxCycles):
        grad = xMat.T * (xMat * weights - yMat) / m
        weights = weights - alpha * grad
    return weights


def logisticAcc(dataSet, method, alpha=0.001, maxCycles=500):
    """
     计算准确率
    :param dataSet: DF数据集
    :param method: 计算权重函数
    :param alpha: 步长
    :param maxCycles: 最大迭代次数
    :return:
         trainAcc: 模型预测准确率
    """

    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = regularize(xMat)
    ws = method(dataSet, alpha=alpha, maxCycles=maxCycles)
    p = sigmoid(xMat * ws).A.flatten()
    for i, j in enumerate(p):
        if j < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    train_error = (np.fabs(yMat.A.flatten() - p)).sum()
    trainAcc = 1 - train_error / yMat.shape[0]
    return trainAcc


print(logisticAcc(dataSet, BGD_LR, alpha=0.01, maxCycles=500))
