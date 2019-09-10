


#matplotlib_animation_sample.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
def f(x1,x2):
    return 0.1*x1**2+x2**2
x1=np.linspace(-10,10,1000)
x2=np.linspace(-10,10,1000)

X1,X2=np.meshgrid(x1,x2)
Z=f(X1,X2)
plt.contour(X1,X2,Z,levels=30)


SGD_xdata, SGD_ydata = [], []
Ada_xdata, Ada_ydata = [], []
ln1,ln2=ax.plot([],[],[],[],animated=False)


plt.contour(X1,X2,Z,levels=30)


def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    global SGD_x1,SGD_x2,Ada_x1,Ada_x2
    SGD_x1=7
    SGD_x2=8
    Ada_x1=7
    Ada_x2=8
    return ln1,ln2
lr_x1=0
lr_x2=0
def update(i): #i is an int from 0 to frames-1, and keep looping
    global x1,x2,SGD_x1,SGD_x2,lr_x1,lr_x2,Ada_x1,Ada_x2


    sgd_g_x1=(0.2*SGD_x1)
    sgd_g_x2=(2*SGD_x2)
    ada_g_x1=(0.2*Ada_x1)
    ada_g_x2=(2*Ada_x2)
    #ada
    lr_x1=lr_x1+ada_g_x1**2
    lr_x2=lr_x2+ada_g_x1**2
    Ada_x1=Ada_x1-1.5/(lr_x1**0.5)*ada_g_x1
    Ada_x2=Ada_x2-1.5/(lr_x2**0.5)*ada_g_x2

    #sgd
    lr=1.5
    SGD_x1=SGD_x1-lr*sgd_g_x1
    SGD_x2=SGD_x2-lr*sgd_g_x2
    SGD_xdata.append(SGD_x1)
    SGD_ydata.append(SGD_x2)
    Ada_xdata.append(Ada_x1)
    Ada_ydata.append(Ada_x2)
    ln1.set_data(SGD_xdata,SGD_ydata)
    ln2.set_data(Ada_xdata,Ada_ydata)
    return ln1,ln2

def main():
    ani = FuncAnimation(fig, update, frames = 500, interval = 100,init_func=init, blit=True)

    plt.show()

if __name__ == '__main__':
    main()
