#-*- coding: utf-8 -*-
#平稳性检验
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#参数初始化

discfile = '../data/discdata_processed.xls'
predictnum =5 #不使用最后5个数据

data = pd.read_excel(discfile,index_col='COLLECTTIME')
del data['SYS_NAME']
print(data.columns)
print(data[[data.columns[1]]])



def plot_data(datav):
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

  ax = datav.plot(marker='o')
  ax.grid(True)
  plt.xticks(datav.index)
  datav.sort_values(by=datav.columns[0])
  ys = np.linspace(datav.values.min(), datav.values.max(), 15)
  plt.yticks(ys.tolist())
  plt.plot()
  # for i in data.index:
  #   # print(i)
  #   b=data[data.columns[1]][i]
  #   # t=i.strftime('%m-%d')
  #   plt.text(i,b,b)
  plt.show()


# datav = data[[data.columns[1]]]
# datav=datav.diff().dropna()
# plot_data(datav)

data = data.iloc[: len(data)-5] #不检测最后5个数据

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
diff = 0
adf = ADF(data['CWXT_DB:184:D:\\'])
while adf[1] > 0.05:
  diff = diff + 1
  adf = ADF(data['CWXT_DB:184:D:\\'].diff(diff).dropna())

print(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(diff, adf[1]))