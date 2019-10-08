import pandas as pd
import numpy as np

dataSet = pd.read_table('testSet.txt', header=None)
dataSet.columns = ['x1', 'x2', 'labels']

print(dataSet)


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
