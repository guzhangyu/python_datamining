import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

data=pd.read_excel('data.xls')
columns=[
'肝气郁结证型系数',
'热毒蕴结证型系数',
'冲任失调证型系数',
'气血两虚证型系数',
'脾胃虚弱证型系数',
'肝肾阴虚证型系数'
]

for a in columns:
    xx=data[a].values
    km=KMeans(n_clusters=4).fit(np.reshape(xx,(len(xx),1)))
    means=pd.DataFrame(km.cluster_centers_,columns=['cc']).sort_values('cc')
    means=means.index.tolist()

    data[a]=[means.index(i) for i in km.labels_]

new_columns=[
'肝气郁结证型系数',
'热毒蕴结证型系数',
'冲任失调证型系数',
'气血两虚证型系数',
'脾胃虚弱证型系数',
'肝肾阴虚证型系数',
'TNM分期']
new_columns1=[
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'H'
    ]
d=data[new_columns]
print(d)
#
# d['TNM分期']=[int(x[1:]) for x in d['TNM分期']]
#     #map(lambda x:int(x[1:]),d['TNM分期'])
# print(d)
# print(d.sum())

from apriori import find_rule
dd=d.as_matrix()
b=[pd.Series(1, index = [new_columns1[i]+str(x[i]) for i in range(0,len(x))]) for x in dd]
find_rule(pd.DataFrame(b).fillna(0),0.06,0.75)
    # b = [0, 0, 0, 0]
    # pre = 0
    # for i in range(3, -1, -1):
    #     b[i] = np.count_nonzero(xx >= means[i]) - sum(b[i:])
    #     now_pre = np.count_nonzero(xx >= km.cluster_centers_[i])
    #     # print(now_pre-pre)
    #     pre = now_pre

        # x1 = set([k for k in range(0, len(km.labels_)) if km.labels_[k] == i])
        # x2 = set([k for k in range(0, len(xx)) if xx[k] >= means[i] and (i == 3 or xx[k] < means[i + 1])])
        # print(i)
        # print(means)
        # print(min(xx[list(x2)]), max(xx[list(x2)]))
        # print(x1 - x2)
        # print(x2 - x1)