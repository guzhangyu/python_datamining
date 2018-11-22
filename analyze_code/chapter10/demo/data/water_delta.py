import pandas as pd
import numpy as np
import datetime

#/Users/zhangyugu/Desktop/heater_
data=pd.read_excel('water_heater.xls')#,usecols=['发生时间'],'开关机状态','保温中','实际温度','热水量','水流量','当前设置温度','加热剩余时间'
data=data[data['水流量']>0]
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format = '%Y%m%d%H%M%S')

def event_num(dt):
    return (data['发生时间'].diff()>dt).sum()+1

deltas=[pd.Timedelta(minutes=i) for i in np.arange(1,9,0.25)]
p=pd.DataFrame(deltas,columns=['时间间隔'])
p['事件数']=p['时间间隔'].apply(event_num)
p['斜率']=p['事件数'].diff()*4
p['斜率指标']=p['斜率'].abs().rolling(4).mean()
print(p)


# result=[]
# for i in deltas:
#     result.append((data['发生时间'].diff()>delta).sum()+1)
#
# # import matplotlib.pyplot as plt
# # # plt.scatter(np.arange(0,10,0.25)[3:],result[3:])
#
#
# print(len(result))
# rates=pd.Series(np.abs(np.diff(result))).rolling(4).mean()*4 #这次少1，不影响
# print(len(rates))
# print(rates)
#
# idx=rates.idxmin()
# print(deltas[1:][idx-3])
# print(rates[idx])
#
# # x=deltas[4:-3]
# # y=rates[3:-3]
# # plt.scatter(x[3:],y[3:])
# # plt.show()