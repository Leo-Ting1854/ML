import matplotlib.pyplot as plt





#2d data -(x,y) 資料處理成矩陣
x_data=[338, 333, 328, 207, 226, 25, 179, 60, 208, 606]#初始數據x 有10筆資料 每筆資料只有1feature
y_data=[640, 633, 619, 393, 428,27, 193, 66, 226, 1591]#初始數據x 有10個對應的label


#因為要使用平方法  如果用numpy 的資料型態會溢位  這代表我們不能用numpy來計算
#這邊用2維
#要找y=wx+b 的w 跟b 使ypredction-y有最小平方值
#隨意選  我們令b=w0 w=w1
w0=0
w1=0
#最小平方=Lossfunction=sigma[((w1x+w0)-rawdatay)**2] 用這個w0 w1
#對10筆資料的所有最小平方法
#Lossfunction的梯度(針對w變數)=sigma(2*(w1+w0-rawdatay)*x)
#Lossfunction的梯度(針對b變數)=sigma(2*(w1x+w0-rawdatay)*1)

#gradientWo=g_wo

lr=10
Iteration=100000
import math
LossUpdate=[]
Time=[]
#梯度下降法演算法
lr_w0=0
lr_w1=1
for time in range(Iteration):
    n=0
    loss=0
    g_w0=0
    g_w1=0

    for n in range(len(x_data)): #計算這個w0 w1 對10筆資料的lossfunction之梯度
        g_w0=g_w0-2*(-w1*x_data[n]-w0+y_data[n])*1
        g_w1=g_w1-2*(-w1*x_data[n]-w0+y_data[n])*x_data[n]
        loss=loss+((w1*x_data[n]+w0-y_data[n])**2)
#    #adagrad
    # lr_w0=lr_w0+g_w0**2
    # lr_w1=lr_w1+g_w1**2
    # w0=w0-lr/(lr_w0**0.5)*g_w0
    # w1=w1-lr/(lr_w1**0.5)*g_w1

    #SGD
    w0=w0-0.000001*g_w0
    w1=w1-0.000001*g_w1

    if time%(Iteration/10)==0:
        LossUpdate.append(loss)
        Time.append(time)
print('最後的梯度=',g_w0,g_w1)
print('最初的最小平方=',LossUpdate[0])
print('最後的最小平方=',LossUpdate[len(LossUpdate)-1])
plt.figure()
plt.plot(Time,LossUpdate)

ypredction=[]
for x in x_data:
    ypredction.append(w1*x+w0)
print('w=',w1,'b=',w0)

plt.figure()
plt.plot(x_data,ypredction)
plt.scatter(x_data,y_data)
plt.title('LinearRegression')
plt.show()
