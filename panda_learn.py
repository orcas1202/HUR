import pandas as pd
from sklearn import datasets
import numpy as np

# pandas.set_option(option, value) 常用操作如下 ------------
pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', None)  # 意为，展示的最多列数 ---> 没有限制
pd.set_option('mode.chained_assignment', None)  # 有待研究
# -----------------------------

iris = datasets.load_iris()


iris_dataFrame = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],   # np,c_连接函数
    columns=iris['feature_names'] + ['target']  # pandas 有待研究
)


print(iris_dataFrame.head())      # 表头的数据.....
print(iris_dataFrame)             # 打印整张表
print(iris_dataFrame.describe())  # 打印整张表的统计数据
