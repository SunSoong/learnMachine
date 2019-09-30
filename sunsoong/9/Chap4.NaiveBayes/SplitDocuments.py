import numpy as np
from functools import reduce


def loadDataSet():
    """
    创建实验数据集
    :return: postingList: 切分好的样本词条
             classVec: 类标签向量
    """
    dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  # 切分好的四条
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量,1代表侮辱性词汇,0代表非侮辱性词汇
    return dataSet, classVec


def createVocabList(dataSet):
    """
    将切分的样本词条整理成词汇表(不重复)
    :param dataSet: 切分好的样本词条
    :return: 不重复的词汇表
    """
    vocabSet = set()  # 创建一个空的集合
    for doc in dataSet:  # 遍历dataSet中的每一条言论
        vocabSet = vocabSet | set(doc)  # 取并集,去重
        vocabList = list(vocabSet)
    print(vocabList)
    return vocabList


def setOfWord2Vec(vocabList, inputSet):
    """
    根据vocabList词汇表,将inputSet向量化,向量额每个元素为1或者0
    :param vocabList: 词汇表
    :param inputSet: 切分好的词条列表中的一条
    :return: 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中,则变为1
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"{word} is not in my vocabulary!")
    return returnVec  # 返回文档向量


def get_trainMat(dataSet):
    """
    生成训练集向量列表
    :param dataSet: 切分好的样本词条
    :return: 所有额词条向量组成的列表
    """
    trainMat = []  # 初始化向量列表
    vocabList = createVocabList(dataSet)  # 生成词汇表
    for inputSet in dataSet:  # 遍历样本词条中每一条样本
        returnVec = setOfWord2Vec(vocabList, inputSet)  # 将当前词条向量化
        trainMat.append(returnVec)  # 追加到向量列表
    print("returnVec: ", returnVec)
    print("trainMat: ", trainMat)
    return trainMat


def trainNB(trainMat, classVec):
    """
    朴素贝叶斯分类器训练函数
    :param trainMat:训练文档矩阵
    :param classVec:训练类别标签向量
    :return: p0v: 非侮辱类的条件概率数组
             p1v: 侮辱类的条件概率数组
             pAb: 文档属于侮辱类的概率
    """
    n = len(trainMat)  # 计算训练的文档数目
    m = len(trainMat[0])  # 计算每篇文档的词条数
    pAb = sum(classVec) / n  # 文档属于侮辱类的概率
    p0Num = np.zeros(m)  # 词条出现数初始化为0
    p1Num = np.zeros(m)  # 词条出现数初始化为0
    p0Denom = 0  # 分母初始化为0
    p1Denom = 0
    for i in range(n):  # 遍历每一个文档
        if classVec[i] == 1:  # 统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1v = p1Num / p1Denom
    p0v = p0Num / p0Denom
    return p0v, p1v, pAb


def classifyNB(vec2Classify, p0v, p1v, pAb):
    """
    朴素贝叶斯分类器分类函数
    :param vec2Classify: 待分类的词条数组
    :param p0v:非侮辱类的条件概率数组
    :param p1v: 侮辱类条件概率数组
    :param pAb: 文档属于侮辱类的概率
    :return:
             0:属于非侮辱类
             1:属于侮辱类
    """
    p1 = reduce(lambda x, y: x * y, vec2Classify * p1v) * pAb
    p0 = reduce(lambda x, y: x * y, vec2Classify * p0v) * (1 - pAb)
    print("p0: ", p0)
    print("p1: ", p1)
    if p1 > p0:
        return 1
    else:
        return 0


def trainingNB(testVec):
    """
    朴素贝叶斯测试函数
    :param testVec:测试样本
    :return:测试样本的类别
    """
    dataSet, classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMat = get_trainMat(dataSet)
    p0v, p1v, pAb = trainNB(trainMat, classVec)
    thisone = setOfWord2Vec(vocabList, testVec)
    if classifyNB(thisone, p0v, p1v, pAb):
        print(testVec, "属于侮辱类")
    else:
        print(testVec, "属于非侮辱类")


dataSet, classVec = loadDataSet()
trainMat = get_trainMat(dataSet)

p0v, p1v, pAb = trainNB(trainMat, classVec)
# 测试样本1
testVec1 = ['love', 'my', 'dalmation']
trainingNB(testVec1)

# 测试样本2
testVec2 = ['stupid', 'garbage']
trainingNB(testVec2)
