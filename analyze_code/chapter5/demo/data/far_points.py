import pandas as pd
import numpy as np

data=pd.read_excel('consumption_data.xls',index_col='Id')
data=(data-data.mean())/data.std()

from sklearn.cluster import KMeans
km=KMeans(n_clusters=3).fit(data)

def dis(a,b):
    return np.linalg.norm(a-b)
    # s=0
    # for i in range(0,3):
    #     s=abs(a[i]-b[i])
    # return s/3

dx=[[],[],[]]
dy=[[],[],[]]
data=data.values
for j in range(0,len(km.labels_)):
    i=km.labels_[j]
    dx[i].append(j)
    dy[i].append(dis(data[j],km.cluster_centers_[i]))

import matplotlib.pyplot as plt
colors=['r','b','g']
for i in range(0,3):
    pdy=pd.DataFrame(dy[i])
    dy[i]=dy[i]/pdy.describe()[0]['50%']
    plt.scatter(dx[i],dy[i],c=colors[i],label=str(i))
    plt.legend()

plt.show()