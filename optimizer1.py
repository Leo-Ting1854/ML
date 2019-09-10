
import  numpy as np
import  matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

fig,ax=plt.subplots()
def f1(x):
    return x**2
def f2(x):
    return x**4-50*x**3-x+1
X=np.linspace(-60,60,10000)
y=f2(X)
plt.plot(X,y)


x1=-40
y1=f2(x1)
x2=-40
y2=f2(x2)
SGD=plt.scatter(x1,y1,c='r',label='SGD')
momentum=plt.scatter(x2,y2,c='b',label='momentum')
ax.set_xlim(-60,60)
ax.set_ylim(-1000000,2500000)

g=4*x2**3-150*x2**2-1
v_last=0.00001*g

def updeate(i):
    global x1,x2,y1,y2,v_last
    g1=4*x1**3-150*x1**2-1
    x1=x1-0.00001*g1
    y1=f2(x1)
    SGD.set_offsets([x1,y1])
    g2=4*x2**3-150*x2**2-1
    v=0.9*v_last+0.00001*g2
    v_last=v
    x2=x2-v
    y2=f2(x2)
    momentum.set_offsets([x2,y2])
    return SGD,momentum

anim=FuncAnimation(fig,updeate,frames=200,interval=200)
plt.legend()#生成標籤
plt.show()
