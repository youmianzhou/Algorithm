from numpy import *
import numpy as np

"""
已有一个user-item评分矩阵,新用户对item的评分矩阵new
求给信用户推荐item

注:这种方法是不对,现实中没有说用户给商品评完分后,你还给他推荐
用户如果没有用过商品,怎么会给商品评分？

一般现在是用协同过滤算法
"""


def cosSim(inA, inB):
    # 余弦相似度计算
    eps = 1.0 * pow(10, -6)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return float(inA * inB.T) / (denom + eps)


def recommend(dataSet, newVest, r=3, rank=1, distCalc=cosSim):
    # 由于可能不是方阵,所以矩阵的行数和列数可能不是一样的
    m, n = np.shape(dataSet)
    limit = min(m, n)

    # 这里的r只取3,这样的矩阵很小,这个算法只是做实验能用而已
    if r > limit:
        r = limit

    # numpy里面可以做svd分解
    # dataSet原本是u-i矩阵,转置后变成i-u矩阵
    U, S, VT = linalg.svd(dataSet.T)

    V = VT.T
    # 取前r列
    Ur = U[:, :r]
    Sr = diag(S)[:r, :r]
    Vr = V[:, :r]
    # 通过svd,近似找到一个维数低的矩阵代替newVest
    newresult = newVest * Ur * linalg.inv(Sr)
    # print(newresult)

    # 计算测试集和训练集每个记录的相似度
    resultarray = array([distCalc(newresult, vi) for vi in Vr])
    # 排序结果降序
    descindx = argsort(-resultarray)[:rank]

    return descindx, resultarray


if __name__ == '__main__':
    # dataSet是ui矩阵,4个用户对6个物品的评分
    dataSet = mat([[5, 5, 3, 0, 5, 5],
                   [5, 0, 4, 0, 4, 4],
                   [0, 3, 0, 5, 4, 5],
                   [5, 4, 3, 3, 5, 5]])

    # 新用户评分
    newVest = mat([[5, 5, 0, 0, 0, 5]])

    descindx, corr = recommend(dataSet, newVest, r=3, rank=1, distCalc=cosSim)

    # 根据计算结果,新用户与user1用户最相似,去掉新用户和user1用户相同的商品,然后把不同的商品推荐给这个新用户
    print(descindx)
    print(corr)
