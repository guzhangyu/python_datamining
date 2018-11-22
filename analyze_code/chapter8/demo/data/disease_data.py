import pandas as pd
from sklearn.cluster import KMeans #导入K均值聚类算法
import numpy as np

data=pd.read_excel('data.xls')
k=4
for i in range(0,6):
    kmodel = KMeans(n_clusters=k)
    a=data.iloc[:,i]
    a=np.reshape(a.values,(-1,1))
    kmodel.fit(a)
    data.iloc[:,i]=kmodel.labels_+1
    data.iloc[:, i]=data.iloc[:, i].apply(lambda j:data.columns[i]+str(j))
print(data)