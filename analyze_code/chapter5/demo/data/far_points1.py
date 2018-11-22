import pandas as pd

data=pd.read_excel('consumption_data.xls',index_col='Id')

from sklearn.cluster import KMeans
km=KMeans(n_clusters=3).fit(data)

# import numpy as np
# data.values-np.apply_along_axis()
# r=pd.concat([data,pd.Series(km.labels_)])
# print()

# import matplotlib.pyplot as plt
# colors=['r','b','g']
# for i in range(0,3):
#     pdy=pd.DataFrame(dy[i])
#     dy[i]=dy[i]/pdy.describe()[0]['50%']
#     plt.scatter(dx[i],dy[i],c=colors[i],label=str(i))
#     plt.legend()
#
# plt.show()