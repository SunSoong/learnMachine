import pandas as pd
import matplotlib.pyplot as plt


def classfiy0(inX, dataSet, k):
    """
    KNN分类器
    :param inX: 需要预测分类的数据集
    :param dataSet: 已知分类标签的数据集(训练集)
    :param k: k近邻中选择距离最近的k个点
    :return: 返回分类结果
    """
    result = []
    dist = list((((dataSet.iloc[:6, 1:3] - inX) ** 2).sum(1)) ** 0.5)
    dist_1 = pd.DataFrame({"dist": dist, "labels": (dataSet.iloc[:6, 3])})
    print(dist_1)
    dr = dist_1.sort_values(by="dist")[:k]
    re = dr.loc[:, "labels"].value_counts()
    result.append(re.index[0])
    return result


rowdata = {"电影名称": ["无问东西", "后来的我们", "前任3", "红海行动", "唐人街探案", "战狼2"],
           "打斗镜头": [1, 5, 12, 108, 112, 115],
           "接吻镜头": [101, 89, 97, 5, 9, 8],
           "电影类型": ["爱情片", "爱情片", "爱情片", "动作片", "动作片", "动作片"]
           }
movie_data = pd.DataFrame(rowdata)
new_data=[24,76]
print(classfiy0(new_data,movie_data,4))


