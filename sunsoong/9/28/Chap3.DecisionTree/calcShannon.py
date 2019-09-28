import numpy as np
import pandas as pd


def calcShannon(dataSet):
    """
    计算香农熵
    :param dataSet: 原始数据集
    :return: 香农熵
    """
    n = dataSet.shape[0]  # 数据集的总行数
    # loc通过行标签索引行数据   iloc通过行号索引行数据
    iset = dataSet.iloc[:, -1].value_counts()  # 标签(最后一列)的所有类别的个数
    p = iset / n  # 每一类标签所占比
    ent = (-p * np.log2(p)).sum()  # 计算信息熵
    return ent


def createDataSet():
    row_data = {"no surfacing": [1, 1, 1, 0, 0],
                "flippers": [1, 1, 0, 1, 1],
                "fish": ["yes", "yes", "no", "no", "no"]}
    dataSet = pd.DataFrame(row_data)
    # print(dataSet.shape)   #(5,3) 5行3列
    return dataSet


# print(createDataSet())
# print(calcShannon(createDataSet()))

# 选择最优的列进行切分
def bestSplit(dataSet):
    baseEnt = calcShannon(dataSet)  # 计算原始熵
    bestGain = 0  # 初始化信息增益
    axis = -1  # 初始化最佳切分列,标签列
    # print(dataSet.shape[0])   # dataSet的行数 即数据集的数据个数
    for i in range(dataSet.shape[-1] - 1):  # 对特征的每一列进行循环
        levels = dataSet.iloc[:, i].value_counts().index  # 提取出当前列的 所有  取值
        ents = 0  # 初始化子节点的信息熵
        for j in levels:  # 对当前列的每一个取值进行循环
            # levels:{"0","1"}   j=1时,选择dataSet中第i列中为1的进行切片
            childSet = dataSet[dataSet.iloc[:, i] == j]  # 某一个子节点的dataframe
            # print("----------")
            # print(childSet)
            # print("----------")
            ent = calcShannon(childSet)  # 计算某一个子节点的信息熵
            ents += (childSet.shape[0] / dataSet.shape[0]) * ent  # 计算当前列的信息熵
            infoGain = baseEnt - ents  # 计算当前列的信息增益
            if (infoGain > bestGain):
                bestGain = infoGain  # 选择最大信息增益
                axis = i
    return axis


def mySplit(dataSet, axis, value):
    """
   按照给定的列切分数据集
    :param dataSet:原始数据集
    :param axis:指定的列索引
    :param value:指定的属性值
    :return:按照指定列索引和属性值切分后的数据集
    """
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return redataSet


def createTree(dataSet):
    featlist = list(dataSet.columns)  # 取数据集中所有的列
    classlist = dataSet.iloc[:, -1].value_counts()  # 获取最后一列类标签
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]  # 递归出口 没有可分的属性 或者每一个树枝下都是相同的标签
    axis = bestSplit(dataSet)  # 确定当前最佳的且分类的索引
    bestfeat = featlist[axis]  # 获取该索引对应的特征
    myTree = {bestfeat: {}}  # 采用字典嵌套的方式存储树的信息
    del featlist[axis]  # 删除当前特征
    valuelist = set(dataSet.iloc[:, axis])  # 提取最佳切分列所有的属性值
    for value in valuelist:  # 对每一个属性值递归建立树
        myTree[bestfeat][value] = createTree(mySplit(dataSet, axis, value))
    return myTree


def classify(inputTree, labels, testVec):
    """
    对一个测试实例进行分类
    :param inputTree:已经生成的决策树
    :param labels:存储选择的最优特征标签
    :param testVec:测试数据列表,顺序对应原数据集
    :return: 分类结果
    """
    firstStr = next(iter(inputTree))  # 获取决策树的第一个节点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = labels.index(firstStr)  # 第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def acc_classify(train, test):
    inputTree = createTree(train)
    labels = list(train.columns)
    result = []
    for i in range(test.shape[0]):
        testVec = test.iloc[i, :-1]
        classLabel = classify(inputTree, labels, testVec)
        result.append(classLabel)
    test["predict"] = result
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()
    print(f"模型预测准确率为{acc}")
    return test


dataSet = createDataSet()
print(dataSet)
ent = calcShannon(dataSet)
print(ent)
axis = bestSplit(dataSet)
print(axis)
redataSet = mySplit(dataSet, 0, 1)
print(redataSet)
myTree = createTree(dataSet)
print(myTree)

# 树的存储
# np.save("myTree.npy", myTree)
# 树的读取
# read_myTree = np.load("myTree.npy", allow_pickle=True).item()
# print(read_myTree)
train = dataSet
test = dataSet.iloc[:3, :]
acc_classify(train, test)
print(test)
