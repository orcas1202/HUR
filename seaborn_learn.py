import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import pandas as pd

iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


# --------------------- 以下为标准化
scaler = Normalizer().fit(x_train)  # the scaler is fitted to the training set
normalized_x_train = scaler.transform(x_train)  # the scaler is applied to the training set
normalized_x_test = scaler.transform(x_test)  # the scaler is applied to the test set
# ----------------------

dictionary_species = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}  # y-target 代表品种， 有setosa，等三种+
iris_dataframe = pd.DataFrame(data=np.c_[x_train, y_train],
                              columns=iris['feature_names'] + ['target'])
before = sns.pairplot(iris_dataframe.replace({'target': dictionary_species}), hue='target')
before.fig.suptitle('data before normarlization', y=1.08)
# plt.savefig(before.png)
iris_dataframe_normalized = pd.DataFrame(data=np.c_[normalized_x_train, y_train],
                                         columns=iris['feature_names'] + ['target'])

after = sns.pairplot(iris_dataframe_normalized.replace({'target': dictionary_species}), hue='target')
after.fig.suptitle('data after noramlization', y=1.08)
# plt.savefig('myplot.png')  ---------------- 保存图片 -------------在pycharm Project
plt.show()
















##
# scaler= Normalizer().fit(x_train) # the scaler is fitted to the training set
# normalized_x_train= scaler.transform(x_train) # the scaler is applied to the training set
# normalized_x_test= scaler.transform(x_test) # the scaler is applied to the test set
# print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')
#
# print('x train before Normalization')
# print(x_train[0:5])
# print('\nx train after Normalization')
# print(normalized_x_train[0:5])
#
#
# # Before
# # View the relationships between variables; color code by species type
# di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0:'Virginica'} # dictionary
#
# before= sns.pairplot(iris_df.replace({'target': di}), hue= 'target')
# before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)
#
# # After
# iris_df_2= pd.DataFrame(data= np.c_[normalized_x_train, y_train],
#                         columns= iris['feature_names'] + ['target'])
# di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
# after= sns.pairplot(iris_df_2.replace({'target':di}), hue= 'target')
# after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)
# # plt.savefig('myplot.png')  ---------------- 保存图片 -------------在pycharm Project
# plt.show()
##
