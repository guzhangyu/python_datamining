import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']

data=pd.read_excel('discretization_data.xls')

k=6
x=data['肝气郁结证型系数'].values
#x.sort()

def plot(k,xs,title):
    fig=plt.figure()
    pre=0
    dks = []
    for i in range(0, k):
        dks.append(range(pre, pre + len(xs[i])))
        pre = pre + len(xs[i])
    ax=fig.add_subplot(111,title=title)
    for i in range(0, k):
        ax.scatter(dks[i], xs[i])

def splitY(x,y):
    ys = []
    pre = y[0]
    for i in y:
        if pre == i:
            continue
        a = x[x <= i]
        a = a[a > pre]
        ys.append(a)
        pre = i
    ys[0] = np.insert(ys[0], 0, x[0])
    return ys

#等宽
dd=data.describe()['肝气郁结证型系数']
ys=splitY(x,np.linspace(dd['min'],dd['max'],k+1))
plot(k, ys,"等宽")

#等频
rk=np.linspace(0,1,k)
cx=x.copy()
cx.sort()
a=np.linspace(0,len(x),k+1).astype(int)-1
a[0]=0
a=cx[a]
print(a)
ys=splitY(x,a)
plot(k, ys,"等频")

#k均值法
from sklearn.cluster import KMeans
import pandas as pd
kmodel=KMeans(n_clusters=k)
kmodel.fit(x.reshape((len(x),1)))

dd=kmodel.cluster_centers_.reshape((1,len(kmodel.cluster_centers_)))[0]
dd=np.sort(dd)
p=pd.DataFrame(dd)
a=p.rolling(2).mean()[0]
a[0]=min(x)
a[k]=max(x)
print(a)

ys=splitY(x,a)
plot(k, ys,"k 均值")


plt.show()


