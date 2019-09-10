import matplotlib.pyplot as plt
import numpy as np
class LogisticRegression():
    def __init__(self):
        self.W=[]
        self.X_train=np.array([])
        self.y_train=np.array([])
        self.iteration=0
    def fit(self,Data_set_x,Data_set_y,iteration_time,Learningrate):
        self.W=np.zeros((Data_set_x.shape[1]+1,)) #建立W=[w0 w1 w2 w3 .....] 維度為X的FEATURES個數(含X0)
        self.X_train=Data_set_x
        self.y_train=Data_set_y
        self.iteration=iteration_time
        self.Learningrate=Learningrate
        self.W=self.adagrad(self.W,self.X_train,self.y_train,self.iteration,self.Learningrate)
        return self.W
    def predict(self,Data_test_x):
        y_predict=self.f_w_b(self.W,Data_test_x)
        for index in range(len(y_predict)):
            if y_predict[index]>0.5:
                y_predict[index]=1
            else:
                y_predict[index]=0
        return  y_predict
    #我希望f_w_b可以接收np.array  然後假設有m筆 資料 每筆資料有n個feature   可接受shape=(m,n)的data集
    #然後 把這資料集的每筆資料經過f_w_b之後的運算值 存入 np.array 回傳 陣列 陣列每格為:每筆資料的f_w_b值
    def f_w_b(self,W,Data_set_x):
        data_number=Data_set_x.shape[0]# 算出總共有幾筆資料
        return self.sigmoid(W,Data_set_x)  #陣列出去

    def sigmoid(self,W,Data_set_x):
        bias_initial=np.array([[1]]*Data_set_x.shape[0]).reshape(Data_set_x.shape[0],1)
        Data_set_new=np.hstack((bias_initial,Data_set_x))

        Z=np.dot(Data_set_new,W)   #做W.T X 內積
        return 1/(1+np.exp(-Z)) #回傳numpy 陣列  每一格代表的是'那筆資料的sigmoid值'

    # # L為每筆資料機率相乘,機率<=1 所以L是0~1的值  又LOG(X)必通過1,0
    # #所以如果L越接近1  -LOG(L)就越接近0   預測效果就越好  反之 如果L很小 0.多  則-LOG(L)會比0大 比較不好
    # #-log(L)=sigma - (y^*Log[f_w_b(資料)]+(1-y^)*log[1-f_w_b(資料)])  定為entropy
    def entropy (self,W,Data_set_x,Data_set_y):#希望log_L接收資料集  回傳資料集的entropy
        entropy_L=0
        f_w_b_result=self.f_w_b(W,Data_set_x)

        for index  in range(len(Data_set_y)):
            if(Data_set_y[index]==0):
                #-log(L)=sigma - (y^*Log[f_w_b(資料)]+(1-y^)*log[1-f_w_b(資料)])  定為entropy
                #乾脆把所有y=0的例子 先把他的f_w_b改成1-f_w_b 這樣就可以把式子改成
                #sigma - (1^*Log[f_w_b(資料)_改動])  定為entropy
                #如果今天遇到Y是Class 2 也就是Y^=0  則把其對應的f_w_b改成1-f_w_b
                f_w_b_result[index]=1-f_w_b_result[index]
        #然後現在f_w_b的各項就變成每筆資料是class1的機率  接著只要利用log 把機率相乘這件事 轉為 相加就好了
        log_f=np.log(f_w_b_result)
        entropy_L=np.sum(-log_f)
        return entropy_L



    def G_vector(self,W,Data_set_x,Data_set_y): #梯度計算器 接收資料集 跟target集 算出 此時的梯度
        data_delta=-(Data_set_y-self.f_w_b(W,Data_set_x)) #回傳陣列 內存 每筆資料的target與f_w_b之差值
        # -( yn^ - f_w_b(Xn) 得到這一項=data_delta
        bias_initial=np.array([[1]]*Data_set_x.shape[0]).reshape(Data_set_x.shape[0],1)
        Data_set_new=np.hstack((bias_initial,Data_set_x))
        G=data_delta[:,np.newaxis]*Data_set_new#製造 含BIAS的 X向量
        #data_delta*Data_set_x=單筆資料差值*資料向量X =[ -( yn^ - f_w_b(Xn) )*xi ]
        #接者由AXIS=0 加總 每一筆資料的[ -( yn^ - f_w_b(Xn) )*xi ]向量值
        G_vector_sum=np.sum(G,axis=0)

        #debug 開啟下面註解
        # print(data_delta) #印出陣列  每列為每筆資料跟f_w_b的差值
        # print(Data_set_new) #印出資料  每列為 x的資料向量
        # print(G) #由 data_delta Data_set_new產生的G
        # print(G_vector_sum) 每列相加後的G

        return G_vector_sum  #回傳gradient向量 是一維列表
    # # gradienti=sigma[ -( yn^ - f_w_b(Xn) )*xi ] # 備註: x0為與bias內積的單位  x0皆為1
    # # 讓Gradient 為向量G內有i個feature  X為向量內有i個feature x0 x1 x2...xi-1
    # # 改寫G=sigam(Xn*X) Xn為每筆資料-( yn^ - f_w_b(Xn)值 X為vector 內有i個feature x0 x1 x2...xi-1
    # # W為weight向量有i個component w0 w1 w2
    # # W=W-lr*G

    #

    def adagrad(self,W,Data_set_x,Data_set_y,iteration_time,Learningrate):#梯度計算器 接收資料集 跟target集 進行迭代找出最佳解W
        #這個函式目的是找出最佳W 使-log(L)最小  利用梯度下降法 不停利用G_vector 去更新W 即可
        #並且利用scale 標準化 優化資料集
        esp=10**-8
        from sklearn.preprocessing import  scale
        Data_set_x_Scale=scale(Data_set_x)
        #利用adagrad 進行學習
        #print(W.shape)  產生跟W一樣維度的gradeintvector加總
        gradeint_history_square=np.zeros((W.shape[0],))
        W_delta=np.zeros((W.shape[0],))

        for time in range(iteration_time):
            g_v=self.G_vector(W,Data_set_x,Data_set_y)
            gradeint_history_square=gradeint_history_square+g_v**2
            W_delta=Learningrate/np.sqrt(gradeint_history_square+esp)*g_v
            W=W-W_delta
            #利用np的向量除法跟sqrt 很方便計算
    #     # a=np.array([1000,2000])
    #     # b=np.array([10,20])
    #     # c=np.sqrt(a/b)
    #     # print(c)
        return  W

    def accuracy_score(self,y_predict,y_data):#y為1維數列
        accurate_number=0
        for index in range(len(y_data)):
            if(y_predict[index]==y_data[index]):
                accurate_number=accurate_number+1
        return accurate_number/len(y_data)

    def noise(self,y_data,nois_rate):#製造noise點
        noise_size=int(nois_rate*len(y_data))
        nois_index=np.random.randint(low=0,high=len(y_data),size=noise_size)
        for index in nois_index:
            if(y_data[index]==1):
                y_data[index]=0
            else:
                y_data[index]=1
        return y_data
    def show_binary_sort(self,Data_set_x,Data_set_y):
        if(Data_set_x.shape[1]==2):
            plt.figure()
            plt.scatter(Data_set_x[:,0],Data_set_x[:,1],c=Data_set_y)
            X=np.linspace(Data_set_x.min()-1,Data_set_x.max()+1,100)
            Y=np.linspace(Data_set_x.min()-1,Data_set_x.max()+1,100)
            x,y=np.meshgrid(X,Y)
            Z=[]
            #自己做Z的meshgrid
            for index2 in range(len(X)):#因為要按造先改變x 再換y的順序
                for index1 in range(len(Y)):
                    Z.append(np.hstack((X[index1],Y[index2])))
            Z=np.array(Z)
            plt.contourf(x,y,self.f_w_b(self.W,Z).reshape(len(X),len(X)),levels=1,alpha=0.2,cmap=plt.cm.hot)
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title("logistic")
            plt.show()
        else:
            print('# WARNING: ')
            print('Only two feature data can use show_binary_sort')
            print('your data feature number is',Data_set_x.shape[1])
# #功能測試區
# X=np.array([[1,2],[3,4]])  #每筆資料有2個變數feature
# Y=np.array([1,0])
# W=np.array([1,0,0])#兩個變數 W=W0 W1 W2  W0為BIAS權重
# A=LogisticRegression()
# W=A.fit(X,Y,100,1)
# print('fit 測試',W)
# print('predict測試',A.predict(X))
# print('f_w_b測試',A.f_w_b(W,X))
# print('entropy測試',A.entropy(W,X,Y)) #相當接近0 很好
# print('accuracy_score測試',A.accuracy_score(A.predict(X),Y))
# A.show_binary_sort(X,Y)#畫圖測試






#entropy
#Z=W.T x X +b  X為一筆資料 內有多個Feature: x0 x1 x2...  其中x0 預設為1
#sigmoid=1/(1+e^-Z)
# f_w_b(X)=sigmoid(Z )=simoid(=W.T x X +b)  如果大於0.5 則為CLASS1  小於0.5 則為CLASS2

#L=f_w_b(資料1)*f_w_b(資料2)*f_w_b(資料3)*........ logistic function  越大代表越有可能從這個MODEL 產生資料1 資料2 資料3.....
#也就是L越大 代表越準確
#L的意思是 最符合這些點的模型  如果今天有兩類class 1 class 2
#其中x_data1=class1;x_data2=class2  然後定義f_w_x()為此點為class1的機率
#最有可能產生這些點的機率為L=f_w_b(x_data1)*(1-f_w_b(x_data2))  因為x_data2是class2 所以
#L產生了 CLASS1的X1 跟CLASS2的X2的機率為l=f_w_b(x_data1)*(1-f_w_b(x_data2))   ,
#(1-f_w_b(x_data2))   ,f_w_b(x_data2)是x_data2是class1的機率  #(1-f_w_b(x_data2))是x_data2是class2的機率

#求L越大越好,但為了使用梯度下降法，求最小就代表 求-LOG(L)越小越好   優點是取LOG可把相乘改相加
#L最大的解跟-log(L)最小的解是同一個意思
#假設資料1 2為CLASS1 資料3為CLASS2
#L=f_w_b(資料1)*f_w_b(資料2)*[1-f_w_b(資料3)]
#LOG(L)=log[f_w_b(資料1)]+log[f_w_b(資料2)]+log[1-f_w_b(資料3)]
#因為有f_w_b()及[1-f_w_b]之分  要把她轉換成單一化
#因為分兩類 只有y_真=1或0  y_真=y^ (y head)
#可以弄成 每筆=y*f_w_b(資料)+(1-y)*[1-f_w_b(資料)] 這樣就可以不用考慮他是0 還是1
#可以改寫成 LOG(L)=y1^*log[f_w_b(資料1)]+(1-y1^)*log[1-f_w_b(資料1)]+y2^*log[f_w_b(資料2)]+(1-y2^)*log[1-f_w_b(資料2)]+....
#代表LOG(L)=sigma(y^*Log[f_w_b(資料)]+(1-y^)*log[1-f_w_b(資料)])
#-log(L)=sigma - (y^*Log[f_w_b(資料)]+(1-y^)*log[1-f_w_b(資料)])
#則目標求 -log(L) 的最小值就是求L的MAX  之WEIGHT
#需要-Log(L)對W的偏微分  即可利用梯度下降法找最佳解

#Wi=Wi-lr*gradienti  這邊的Wi就是 Weight的第i個component  gradienti就是gradient的第i個component
#gradienti=sigma[ -( yn^ - f_w_b(Xn) )*xi ]
#gradient的第i個component=Total[ [1筆資料真實-1筆資料預測]*資料的第i個feature ]
#gradient的第i個component=sigma(單筆資料差*Xi)
#讓Gradient 為向量G內有i個feature  X為向量內有i個feature x0 x1 x2...xi-1
#改寫G=sigam(Xn*X) Xn為每筆資料  的 -( yn^ - f_w_b(Xn)值
#W為weight向量有i個component w0 w1 w2
#W=W-lr*G
#這邊lr再看看要使用何種優化演算法 也許為adagrade


#我希望f_w_b可以接收np.array  然後假設有m筆 資料 每筆資料有n個feature   可接受shape=(m,n)的data集
#然後 把這資料集的每筆資料經過f_w_b之後的運算值 存入 np.array 回傳 陣列 陣列每格為:每筆資料的f_w_b值
