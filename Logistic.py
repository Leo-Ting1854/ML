import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split  #分開模塊:
#Split arrays or matrices into random train and test subsets

import ML_model as ML #自建模組
from sklearn import  datasets
from sklearn.model_selection import train_test_split  #分開模塊:
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.2)
    return X, y

# iris=datasets.load_iris()
# iris_x=iris.data[0:100,[1,2]]#這個例子可以調1~4個feature 改0:x就好
# iris_y=iris.target[0:100]
testsize=0.3
# #iris資料
# x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=testsize,random_state=42)

#moon資料
X,y=generate_data()
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=testsize)
LR=ML.LogisticRegression()
#參數


iteration_time=1000
intial_learningrate=1
# print('nois_rate=',noise_rate,'iteration_time=',iteration_time,'intial_learningrate=',intial_learningrate)
#
# y_train=LR.noise(y_train,noise_rate)
W=LR.fit(x_train,y_train,iteration_time,intial_learningrate)
y_predict=LR.predict(x_train)
print('訓練資料上準確度=',LR.accuracy_score(y_predict,y_train))
LR.show_binary_sort(x_test,y_test)

y_predict=LR.predict(x_test)
print('在測試資料上準確度=',LR.accuracy_score(y_predict,y_test))
LR.show_binary_sort(x_test,y_test)
