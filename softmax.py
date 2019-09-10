import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
class softmax_classification:
    def __init__(self):
        self.Weight=[]
        self.Bias=[]
        self.adagrad_Weight=0
        self.adagrad_Bias=0
        self.optimizer_name="adagrad"
    def softmax(self,A):
        prob=np.copy(A)*0

        for data_index in range(len(A)):
            max=np.max(A[data_index])
            down=np.sum(np.exp(A[data_index]-max))
            for A_index in range (len(A[data_index])):
                prob[data_index][A_index]=np.exp(A[data_index][A_index]-max)/down
        return prob
    def Loss(self,Y_prob,y):
        return -np.sum(y*np.log(Y_prob))

    def process_Y(self,kind,y):
        Y_process=np.zeros((len(y),len(kind)))
        for data_index in range(len(y)):
            for sort in kind:
                if y[data_index]==sort:
                    Y_process[data_index][sort]=1
        return Y_process
    def fit(self,X,y,optimizer_name,lr,iter_time):
        self.optimizer_name=optimizer_name
        kind=np.unique(y)
        Weight=np.random.rand(X.shape[1],len(kind))
        Bias=np.random.rand(len(kind))
        Y_process=self.process_Y(kind,y)
        Loss_line_y=[]
        Loss_line_x=[]
        for time in range(iter_time):
            A=np.dot(X,Weight)+Bias
            Y_prob=self.softmax(A)
            partial_L_Z=Y_prob-Y_process
            W_gradient=np.dot(X.T,partial_L_Z)
            Bias_gradient=np.sum(np.copy(partial_L_Z),axis=0)
            Weight,Bias=self.optimizer(Weight,W_gradient,Bias,Bias_gradient,lr)
            Loss=self.Loss(Y_prob,Y_process)
            if(time%(iter_time/10)==0):
                print('time=',time,'Loss=',Loss)
                Loss_line_x.append(time)
                Loss_line_y.append(Loss)
        plt.figure()
        plt.title("loss_line")
        plt.xlabel("time")
        plt.ylabel("loss")
        plt.plot(Loss_line_x,Loss_line_y,label="loss_line")
        plt.legend(loc="upper right")
        self.Weight=Weight
        self.Bias=Bias
    def show(self,X,y):
        fig=plt.figure()#產生視窗
        plt.title("dot_decision")
        kind=np.unique(y)
        if(X.shape[1]==2):
            # XX=np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,len(X))
            plt.scatter(X[:,0],X[:,1],c=y)
            #法1
            #機率的演算法不能用這樣的線方法來顯示 因為沒辦法考慮z
            # for index in range(len(self.Weight.T)):
            #     w1,w2=self.Weight.T[index][0],self.Weight.T[index][1]
            #     b=self.Bias[index]
            #     Y=-(XX*w1+b)/w2
            #     plt.plot(XX,Y)

            ##法2
            x_min,x_max=X[:,0].min(),X[:,0].max()
            y_min,y_max=X[:,1].min(),X[:,1].max()
            X_point=np.linspace(x_min-0.5,x_max+0.5,100)
            Y_point=np.linspace(y_min-0.5,y_max+0.5,100)
            X,Y=np.meshgrid(X_point,Y_point)#座標點
            #然後我們要把這些X,Y 照著座標組合成一筆新的X_test
            #ravel可以很好的幫我把兩個座標點按照順序轉成1維
            #然後我就可以np.c_[] 重疊組合 生成新的X_TEST=(x,y) 2d點
            X_=np.c_[X.ravel(),Y.ravel()]
            z_prob,z_predict=self.predict(X_)#但此時還是2維組成的序列
            z_predict=z_predict.reshape(X.shape)

            plt.contour(X,Y,z_predict,levels=10)

            # #法3
            # x_min,x_max=X[:,0].min(),X[:,0].max()
            # y_min,y_max=X[:,1].min(),X[:,1].max()
            # X_point=np.linspace(x_min-0.5,x_max+0.5,10)
            # Y_point=np.linspace(y_min-0.5,y_max+0.5,10)
            # X,Y=np.meshgrid(X_point,Y_point)#座標點
            # X_=np.c_[X.ravel(),Y.ravel()]
            # z_prob,z_predict=self.predict(X_)#但此時還是2維組成的序列
            #
            # z_prob=z_prob.reshape(len(X_point),len(Y_point),len(kind))
            # z_predict=z_predict.reshape(len(X_point),len(Y_point))
            # X_=X_.reshape(len(X_point),len(Y_point),2)
            # #Line1
            # Y1_plot_point=[]
            # x1_plot_point=[]
            # #Line2
            # Y2_plot_point=[]
            # x2_plot_point=[]
            # #Line3
            # Y3_plot_point=[]
            # x3_plot_point=[]
            # for X_index in range(len(X)):
            #     done_flag=0
            #     for y_index in range(len(Y)):
            #         if(z_predict[y_index][X_index]==0):
            #             Y1_plot_point.append(X_[y_index][X_index][1])
            #             x1_plot_point.append(X_[y_index][X_index][0])
            #             done_flag=1
            #         elif(z_predict[y_index][X_index]==1):
            #             Y2_plot_point.append(X_[y_index][X_index][1])
            #             x2_plot_point.append(X_[y_index][X_index][0])
            #             done_flag=1
            #         elif(z_predict[y_index][X_index]==2):
            #             Y3_plot_point.append(X_[y_index][X_index][1])
            #             x3_plot_point.append(X_[y_index][X_index][0])
            #             done_flag=1
            # plt.plot(x1_plot_point,Y1_plot_point,color="red")
            # plt.plot(x2_plot_point,Y2_plot_point,color="blue")
            # plt.plot(x3_plot_point,Y3_plot_point,color="green")


        elif(X.shape[1]==3):
            ax=Axes3D(fig)#產生3D 軸
            XX=np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,len(X))
            YY=np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,len(X))
            X_mesh,Y_mesh=np.meshgrid(XX,YY)
            ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
            for index in range(len(self.Weight.T)):
                #暫時找不到替代方案  只好用感知器的線性分割  肯定不太準
                w1,w2,w3=self.Weight.T[index][0],self.Weight.T[index][1],self.Weight.T[index][2]
                b=self.Bias[index]
                Z=-(X_mesh*w1+Y_mesh*w2+b)/w3
                ax.plot_surface(X_mesh,Y_mesh,Z)

    def optimizer(self,Weight,W_gradient,Bias,Bias_gradient,lr):
        if self.optimizer_name=="adagrad":
            self.adagrad_Bias+=Bias_gradient**2
            self.adagrad_Weight+=W_gradient**2
            Weight=Weight-lr/(np.sqrt(self.adagrad_Weight)+10**-8)*W_gradient
            Bias=Bias-lr/(np.sqrt(self.adagrad_Bias)+10**-8)*Bias_gradient
        if self.optimizer_name=="SGD":
            Weight=Weight-lr*W_gradient
            Bias=Bias-lr*Bias_gradient

        return Weight,Bias
    def predict(self,X):
        Y_prob=self.softmax(np.dot(X,self.Weight)+self.Bias)
        y_predict=Y_prob.argmax(axis=1)
        return Y_prob,y_predict
    def accuaracy(self,y_predict,y):
        num=0
        for data_index in range(len(y)):
            if(y[data_index]==y_predict[data_index]):
                num+=1
        return num/len(y)*100



def main():
    from sklearn import  datasets
    #X,y=datasets.make_blobs(300,n_features=2)

    #線性不可分 肯定悲劇
    X,y=datasets.make_moons(200)
    from sklearn.preprocessing import scale
    digits=datasets.load_digits()
    X=digits.data
    y=digits.target
    X=scale(X)

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    S=softmax_classification()
    S.fit(x_train,y_train,"adagrad",1,iter_time=200)

    y_prob,y_predict=S.predict(x_train)
    print("訓練資料準確度=",S.accuaracy(y_predict,y_train))
    if(1<X.shape[1]<4):
        S.show(x_train,y_train)
        plt.title("train")

    y_prob,y_predict=S.predict(x_test)
    print("測試資料準確度=",S.accuaracy(y_predict,y_test))
    if(1<X.shape[1]<4):
        S.show(x_test,y_test)
        plt.title("test")

    plt.show()

    # #設定框框大小 (寬,高)
    # fig=plt.figure(figsize=(10,5))#主框框
    #
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)#high  wide
    # #
    #
    # # 把前 8 個手寫數字顯示在子圖形
    # for i in range(10):
    #     # 在 2 x 5 網格中第 i + 1 個位置繪製子圖形，並且關掉座標軸刻度
    #     ax = fig.add_subplot(2,5, i + 1, xticks = [], yticks = [])
    #     # 顯示圖形，色彩選擇灰階
    #     ax.imshow(digits.images[i], cmap = plt.cm.binary)
    #     # 在左下角標示目標值
    #     ax.text(5,0, 'training'+str(digits.target[i]))
    # # 顯示圖形
    # plt.show()

if __name__=="__main__":
    main()
