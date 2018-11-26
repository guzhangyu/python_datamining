import random
import numpy as np

def distance(a,b):
    if type(a)==list:
        return sum([(b[i]-a[i])**2 for i in range(0,len(a))])**0.5
    return b-a

def fit(data_pre,n_clusters):
    starts=set()

    data=data_pre
        #np.array(data_pre)
    while len(starts)<n_clusters:
        starts.add(random.randint(0,len(data)-1))

    _cluster_centers = []
    _labels=[-1 for i in data]
    k=0
    for i in starts:
        _cluster_centers.append(data[i])
        _labels[i]=k
        k=k+1

    while -1 in _labels:
        # print(_cluster_centers)
        dis=None
        ks=None
        k=None
        for i in range(0,len(data)):
            if _labels[i]!=-1:
                continue

            a=data[i]
            ds=[distance(a,c) for c in  _cluster_centers]
            d=min(ds)
            if dis is None or dis>d:
                dis=d
                ks=[i]
                k=ds.index(d)
            elif dis==d:
                ks.append(i)

        s=sum([1 for i in _labels if i == k])
        _cluster_centers[k]=[(_cluster_centers[k][i]*s+data[ks[0]][i]*len(ks))/(s+len(ks)) for i in range(0,len(_cluster_centers[k]))]
        for i in ks:
            _labels[i] = k
    return _cluster_centers,_labels

# data=[1,3,6,7,8,9,11,150]
data=[[1,3],[6,7],[8,9],[11,150]]
a=fit(data,2)
print(a)
