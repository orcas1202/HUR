from sklearn import datasets
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(iris_X, iris_Y, test_size=0.1)

X_setosa, y_setosa = iris_X[0:50], iris_Y[0:50]            # seotsa\versicolor\virginica 为品种名称
X_versicolor, y_versicolor = iris_X[50:100], iris_Y[50:100]
X_virginica, y_virginica = iris_X[100:150], iris_Y[100:150]

plt.scatter(X_setosa[:, 0], X_setosa[:, 2], color='red', marker='o', label='setosa')
plt.scatter(X_versicolor[:, 0], X_versicolor[:, 2], color="blue", marker="^", label='vericolor')
plt.scatter(X_virginica[:, 0], X_virginica[:, 2], color='brown', marker='s', label='virginica')
plt.xlabel('sepal length')          # 花萼长度
plt.ylabel('petal length')          # 花瓣长度
plt.legend(loc='upper left')
plt.savefig("matplt.jpg")           # 存图至Pycharm Project文件夹
# plt.show()
