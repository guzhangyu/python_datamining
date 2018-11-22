#-*- coding: utf-8 -*-
#用水事件划分
import pandas as pd
import numpy as np

threshold = pd.Timedelta('4 min') #阈值为分钟
inputfile = '../data/water_heater.xls' #输入数据路径,需要使用Excel格式
outputfile = '../tmp/dividsequence1.xls' #输出数据路径,需要使用Excel格式

data = pd.read_excel(inputfile)
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format = '%Y%m%d%H%M%S')
#如果因为采样的原因，导致每两条之间的记录时间就已经大于4分钟，那么可能这两次之间就是一直在用水
data = data[data[u'水流量'] > 0] #只要流量大于0的记录
d = data[u'发生时间'].diff() > threshold #相邻时间作差分，比较是否大于阈值

# print("hah")
# print(d.shape,d.cumsum().shape,(d-d.cumsum()-1).shape)
# test=pd.concat([d,d.cumsum(),d-d.cumsum()-1],axis=1)
# print(test.shape)
# print(test)
# print(d.cumsum() + 1)
data[u'事件编号'] = d.cumsum() + 1 #通过累积求和的方式为事件编号
# print(data[u'事件编号'])
id=np.diff(d.index)

#相邻两次的统计可能时间就已经超过阈值
print(len(data))
aa=list(map(lambda a:a.seconds,data[u'发生时间'].diff()))
del aa[0]
print(id*2<aa)

# writer = pd.ExcelWriter(outputfile)
# data.to_excel(writer)
data.to_excel(outputfile)