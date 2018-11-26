import pandas as pd
data=pd.read_excel('../data/standardized.xls',index_col='基站编号')

from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
# from sklearn.cluster import AgglomerativeClustering
# model=AgglomerativeClustering(n_clusters=3,linkage='ward')
model.fit(data)
data['聚类类别']=pd.Series(model.labels_,index=data.index)
# print(data)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
for i in range(0,3):
    plt.figure()
    plt.title('类别'+str(i))
    for j in data.loc[data['聚类类别']==i].values:
        # print(j)
        plt.plot(range(1,5),j[:4])
    plt.xticks(range(1, 5), ['工作日人均停留时间', '凌晨人均停留时间', '周末人均停留时间','日均人流量'], rotation=20)
    plt.subplots_adjust(bottom=0.15) #调整底部
plt.show()

