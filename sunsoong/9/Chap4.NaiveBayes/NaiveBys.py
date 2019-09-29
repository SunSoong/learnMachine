"""
总结:
    一般来说,如果样本特征的分布大部分是连续性,使用GaussianNB
    如果样本也正的分布大部分是多元离散值,使用MultinomialNB
    如果是二元离散值或者很稀疏的多元离散值,使用BernoulliNB

    手写GaussianNB对鸢尾花数据集进行分类
"""
# 1.导入数据集
import numpy as np
import pandas as pd
import random

dataSet = pd.read_csv("iris.txt", header=None)


# print(dataSet.head())


# 2.切分训练集和测试集
def randSplit(dataSet, rate):
    """
    随机对数据集切分为训练集和测试集
    :param dataSet: 输入的原始书记
    :param rate: 训练集所占的比例
    :return: 切分好的数据集和训练集
    """
    l = list(dataSet.index)  # 提取索引
    random.shuffle(l)  # 随机打乱索引 只打乱索引 不改变运行方式
    dataSet.index = l  # 将打乱后的索引重新复制给原始数据
    n = dataSet.shape[0]  # 总行数
    m = int(n * rate)  # 训练集的行数
    train = dataSet.loc[range(m), :]  # 提取前m个记录作为训练集
    test = dataSet.loc[range(m, n), :]  # 剩下的作为测试集
    dataSet.index = range(dataSet.shape[0])  # 更新原始数据集的索引
    test.index = range(test.shape[0])  # 更新原始测试集的索引
    return train, test


def gnb_classify(train, test):
    # 提取训练集的标签种类  .value_counts 用于计数键值对中每个值得个数 返回列表  .index求列表中的键  即去重后的train的值
    # 得到鸢尾花的三种类型
    labels = train.iloc[:, -1].value_counts().index
    # 运用正态分布求当前测试样本属于哪一类的概率  需要用到均值和方差
    mean = []  # 存放每个类别的均值
    std = []  # 存放每个类别的方差
    result = []  # 存放测试集的预测结果
    for i in labels:
        # print(labels)
        item = train.loc[train.iloc[:, -1] == i, :]  # 对train进行遍历  找到每一个标签等于i的行 提取出来 一共三个标签 四列
        m = item.iloc[:, :-1].mean()  # 求标签的平均值 四个特征 三个标签 一共12个数据
        s = np.sum((item.iloc[:, :-1] - m) ** 2) / (item.shape[0])  # 当前类别的方差
        mean.append(m)  # 将当前类别的平均值追加至列表
        std.append(s)  # 将当前类别的方差追加至列表
    means = pd.DataFrame(mean, index=labels)  # 变为DF格式,索引为类标签
    stds = pd.DataFrame(std, index=labels)  # 变为DF格式 索引为类标签
    print(means)
    print(stds)
    for j in range(test.shape[0]):  # 遍历所有测试集
        iset = test.iloc[j, :-1].tolist()  # 当前测试实例 去除labels后的数据集
        # 正态分布公式 测试目标属于每一种标签的概率
        iprob = np.exp(-1 * (iset - means) ** 2 / (stds * 2)) / (np.sqrt(2 * np.pi * stds))
        prob = 1  # 初始化当前实例总概率
        # 朴素贝叶斯的想法:求出4个特征的概率,然后乘积  得到总概率 (test.shape[1]-1)=4
        for k in range(test.shape[1] - 1):  # 遍历每个特征
            prob *= iprob[k]  # 特征概率之积 为当前实例总概率
            cla = prob.index[np.argmax(prob.values)]  # 返回最大概率的类别 argmax返回最大值索引的值,结果为标签
        result.append(cla)
    test["predict"] = result
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()  # 计算预测准确率
    # print(f"模型预测准确率为{acc}")
    return test


train, test = randSplit(dataSet, 0.8)
gnb_classify(train, test)
