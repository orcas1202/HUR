import numpy as np
import pandas as pd
from collections import Counter


def my_k_neighbour(x_train, y_train, x_test, K):
    """
    ---------------
    K —— 近邻算法
    现有一个鸢尾花标本，根据它的 花瓣，花萼大小，我们采取 该算法预测是什么品种

    原理：对于一个样本，我们找花萼，花瓣大小跟他最相似的那几个标本，
    如果这几个标本是品种A，我们就可以说，手上的这个，有可能是某某品种

    例子：   1. 鸢尾花有4个属性，花萼的长度，宽度，花瓣的长度宽度；
            2. 鸢尾花有3个品种 ：setosa; Versicolour; Virginica
            3. 想象一个四维空间，代表花萼等大小，拿到一个样本，哪一个历史数据和此样本距离最近；我就猜测他是什么品种
               （这是k == 1 时的情况）
            4. 如果 k == 3，我就找三个和他最接近的，那个品种多，选哪个
    实现： a.  算距离
          b.  找最近的K个邻居
          c.  看邻居种类，确定样本是何种类
    ----------------- 以下为代码
    """

    y_pred = []
    for x_test_point in x_test:
        distance_point = distance_ecu(x_train, x_test_point)  # a.算出距离
        df_nearest_point = nearest_neighbors(distance_point, K)  # b. 找到邻居
        y_pred_point = voting(df_nearest_point, y_train)  # c
        y_pred.append(y_pred_point)

    return y_pred


def distance_ecu(x_train, x_test_point):
    """

    ------------
    计算距离有很多种方法，称为"某范式"
    这里是"二范式"，是集合距离，欧几里得距离...
    这里有待研究。

    """
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0

        for column in range(len(current_train_point)):
            current_distance += (current_train_point[column] - x_test_point[column]) ** 2

        # 以上求出了，各个属性的距离之和（平方）
        current_distance = np.sqrt(current_distance)
        distances.append(current_distance)

    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


def nearest_neighbors(distance_point, k):
    #  按照距离，排序整个所有的样本
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)
    df_nearest = df_nearest[:k]
    return df_nearest


def voting(df_nearest, y_train):
    counter_vote = Counter(y_train[df_nearest.index])
    y_pred = counter_vote.most_common()[0][0]  # Majority Voting

    return y_pred



