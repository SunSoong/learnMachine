import numpy as np
import pandas as pd


def createDataSet():
    row_data = {"no surfacing": [1, 1, 1, 0, 0],
                "flippers": [1, 1, 0, 1, 1],
                "fish": ["yes", "yes", "no", "no", "no"]}
    dataSet = pd.DataFrame(row_data)
    return dataSet


def calcShannon(dataSet):
    """
    计算香农熵
    :param dataSet: 原始数据集
    :return: 返回香农熵
    """
    # 先求总行数 即数据个数
    n = dataSet.shape[0]
    # 在计算标签列的各个元素的个数
    iset = dataSet.iloc[:, -1].value_counts()
    # 各个元素所占比例
    p = iset / n
    # 利用公式求熵
    ents = (-p * np.log2(p)).sum()
    return ents


def bestSplit(dataSet):
    """
    选择最优的切分方式
    :param dataSet: 原始数据集
    :return: 最优切分方式的序列
    """
    # 原始香农熵
    baseEnt = calcShannon(dataSet)
    bestGain = 0  # 初始化信息增益
    axis = -1  # 初始化最优切分列
    for i in range(dataSet.shape[-1] - 1):  # 循环遍历原始集的每一列属性
        levels = dataSet.iloc[:, -1].value_counts.index  # 得到当前列的属性的所有值
        ents = 0  # 熵
        # levels:{"0","1"}
        for j in levels:  # 遍历每一个属性
            childSet = dataSet[dataSet.iloc[:, i] == j]  # 取dataSet中每一个 第i列的值=j 的行 即 切分后属于同一分类的集合的元素
            # 计算当前j值 得熵
            ent = calcShannon(childSet)  # 求 当前集合的 熵
            ent += (childSet.shape[0] / dataSet.shape[0]) * ent  # 求 子集合 的信息熵之和
            infoGain = baseEnt - ents  # 求划分后的信息增益
            if (infoGain > bestGain):
                bestGain = infoGain  # 选择最大信息增益
                axis = i
    return axis


def muSplit(dataSet, axis, value):
    """
    按照给定的列 下标切分数据集
    :param dataSet: 原始数据集
    :param axis:指定的列索引
    :param value:指定索引的值
    :return: 按照指定索引划分后的数据集
    """
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return redataSet


dataSet = createDataSet()
print(dataSet)
ents = calcShannon(dataSet)
print(ents)
