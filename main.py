import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from  K_Nearest_Neighbour import my_k_neighbour
from sklearn.preprocessing import Normalizer
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# import iris dataset
iris = datasets.load_iris()

# np.c_ is the numpy 是numpy的函数，连接
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2,   # 测试占总体的比例
                                                    shuffle=True,    # 是否打乱
                                                    random_state=0)  # 初始状态，如果初始值给定一样，那么打乱出的结果也一样
                                                                     # 倘若不给定初始值，每次打乱结果随机

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

K = 3  # 这里K可以通过 K_fold 算法来找到
y_prediction= my_k_neighbour(x_train, y_train, x_test, K)
y_prediction = np.asarray(y_prediction)

print("我对KNN模型的预测结果")
print(y_prediction)
print("\n\n")
print("实际上的结果")
print(y_test)
print("\n\n")