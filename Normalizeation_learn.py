from sklearn import datasets
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y,
                                                    test_size=0.2,  # 规定训练，测试的比列
                                                    random_state=0,  # 随机状态是一个初始值，用它来打乱数据...
                                                    shuffle=True)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')
# 等等再查.shape

scaler = Normalizer().fit(x_train)
x_train_normalized = scaler.transform(x_train)  # 范数，有待研究
x_test_normalized = scaler.transform(x_test)

print('原始的训练数据：')
print(x_train[:5])
print('\n\n')
print('标准化的数据')
print(x_train_normalized[:5])