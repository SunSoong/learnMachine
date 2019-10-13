import pandas as pd
import matplotlib.pyplot as plt

datingTest = pd.read_table("datingTestSet.txt", header=None)  # 导入数据集
# 2.分析数据 matplotlib画散点图
# 每周获取的飞行常客里程数 玩游戏所占的时间比 每周所消费的冰淇淋公升数 喜欢程度
Colors = []
for i in range(datingTest.shape[0]):
    m = datingTest.iloc[i, -1]
    if m == "didntLike":
        Colors.append("black")
    if m == "smallDoses":
        Colors.append("orange")
    if m == "largeDoses":
        Colors.append("red")
# 绘制两两特征之间的散点图
plt.rcParams["font.sans-serif"] = ["Simhei"]  # 图中字体设置为黑体
pl = plt.figure(figsize=(12, 8))  # 画布

fig1 = pl.add_subplot(221)  # 子画布 2行2列中的第一个
# 切玩游戏所占的时间比:datingTest.iloc[:,1]
plt.scatter(datingTest.iloc[:, 1], datingTest.iloc[:, 2], marker=".", c=Colors)
plt.xlabel("玩游戏视频所占时间比")
plt.ylabel("每周消费冰淇淋公升数")

fig2 = pl.add_subplot(222)
plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 1], marker=".", c=Colors)
plt.xlabel("每年飞行常客里程数")
plt.ylabel("玩游戏视频所占时间比")

fig2 = pl.add_subplot(223)
plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 2], marker=".", c=Colors)
plt.xlabel("每年飞行常客里程数")
plt.ylabel("每周消费冰淇淋公升数")

plt.show()


# 数据归一化
def minmax(dataSet):
    minDf = dataSet.min()
    maxDf = dataSet.max()
    normSet = (dataSet - minDf) / (maxDf - minDf)
    return normSet


# minmax(datingTest.iloc[:,:3]:取所有行 第一,二,三列的元素(除了标签列),最后把第四列的元素拼接回来
datingT = pd.concat([minmax(datingTest.iloc[:, :3]), datingTest.iloc[:, 3]], axis=1)


# print(datingT.head())

# 切分训练集和测试集
def randSplit(dataSet, rate=0.9):
    """
    切分训练集和测试集
    :param dataSet:原始数据集
    :param rate:训练集所占比例
    :return: 切分好的训练集:train和数据集:test
    """
    n = dataSet.shape[0]
    m = int(n * rate)
    train = dataSet.iloc[:m, :]
    test = dataSet.iloc[m:, :]
    test.index = range(test.shape[0])
    return train, test


train, test = randSplit(datingT)


# print(train.head())
# print(test.shape[0])

def datingClass(train, test, k):
    """
    KNN-近邻算法分类器
    :param train: 训练集
    :param test: 测试集
    :param k: k-近邻参数,即选择最小的k个点
    :return: 预测好分类的测试集
    """
    n = train.shape[1] - 1  # 训练集 标签之外的列数
    m = test.shape[0]  # 测试集行数
    result = []
    for i in range(m):
        # 每一行(一次数据)都计算一次距离
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) ** 2).sum(1)) ** 0.5)
        dist_l = pd.DataFrame({"dist": dist, "labels": (train.iloc[:, n])})
        dr = dist_l.sort_values(by="dist")[:k]
        re = dr.loc[:, "labels"].value_counts()
        result.append(re.index[0])

    result = pd.Series(result)
    # 添加新的一列
    test["predict"] = result
    # test的倒数第一列和倒数第二列是否一样
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()
    print(f"模型的测试准确率为{acc}")
    return test


datingClass(train, test, 6)
