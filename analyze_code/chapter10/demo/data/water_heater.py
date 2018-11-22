import pandas as pd
import numpy as np
import datetime
from IPython.core.interactiveshell import InteractiveShell

#/Users/zhangyugu/Desktop/heater_
pre_data=pd.read_excel('water_heater.xls')#,usecols=['发生时间'],'开关机状态','保温中','实际温度','热水量','水流量','当前设置温度','加热剩余时间'
pre_data=pre_data[['发生时间', '开关机状态', '保温中', '实际温度', '热水量', '水流量', '当前设置温度', '加热剩余时间']]

# data=data.reindex(np.arange(0,len(data))[data['开关机状态']=='开'])
# print(pre_data)
del pre_data['保温中']
del pre_data['开关机状态']
# data['保温中']=list(map(lambda x : 1 if x=='开' else 0,data['保温中']))
pre_data['实际温度']=list(map(lambda x:int(x[:-3]), pre_data['实际温度']))
pre_data['当前设置温度']=list(map(lambda x:int(x[:-3]), pre_data['当前设置温度']))
pre_data['热水量']=list(map(lambda x:float(x.strip('%'))/100.0, pre_data['热水量']))
pre_data['加热剩余时间']=list(map(lambda x:int(x[:-3]) if x[:-3] else 0, pre_data['加热剩余时间']))
pre_data['发生时间']=list(map(lambda x:datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S'), pre_data['发生时间']))

timeDiff=pre_data['发生时间'].diff()
indexs=timeDiff[timeDiff>pd.Timedelta(seconds=2)].index-1
inserted=[
    [pre_data['发生时间'][i]+pd.Timedelta(seconds=2),0,0,0,0,0]
    for i in indexs
]
pre_data.append(pd.DataFrame(inserted,columns=['发生时间', '实际温度', '热水量', '水流量', '当前设置温度', '加热剩余时间']))
pre_data.sort_index()

data=pre_data[pre_data['水流量']>0]
# print(data)
deltas=[pd.Timedelta(minutes=i) for i in np.arange(1,9,0.25)]
time_diffs=data['发生时间'].diff()
result=[(time_diffs>i).sum()+1 for i in deltas]

import matplotlib.pyplot as plt
# plt.scatter(np.arange(0,10,0.25)[3:],result[3:])


# print(pre_data)
# print(len(result))
rates=pd.Series(np.abs(np.diff(result))).rolling(4).mean()*4 #这次少1，不影响
# print(len(rates))
# print(rates)

idx=rates.idxmin()
print(deltas[idx-3])
print(rates[idx])


# total_uses=list(map(lambda a:pd.Timedelta(a),time_diffs[time_diffs>deltas[idx-3]])) #总用水时间
exceeds=time_diffs[time_diffs>deltas[idx-3]]
# print(exceeds)


# ends=data['发生时间'][exceeds.index]
# uses=[[
#     for j in range(data['发生时间'][i]-exceeds[i],data['发生时间'])
# ] for i in exceeds.index]
uses=[]
total_uses=[]
indexs=[]
pre_index=0
used_quantities=[]
hot_waters=[]
data_index=data.index.values.tolist()
stop_counts=[]
stop_time_waves=[]
flow_waves=[]
def get_gap(j):
    return (pre_data['发生时间'][int(j) + 1] - pre_data['发生时间'][int(j) - 1]) / 2
for i in exceeds.index:
    if i > exceeds.index[0]:
        start=pre_index
    else:
        start = data.index[0] if data.index[0]<i else 1
        #
    # j=i-1
    # use=pd.Timedelta(minutes=0)
    # while pre_data['发生时间'][j]>=pre_data['发生时间'][i]-exceeds[i]:
    #     if pre_data['水流量'][j]>0:
    #         use=use+(pre_data['发生时间'][j+1]-pre_data['发生时间'][j-1])/2
    #     j=j-1
    # a=[]
    # for j in range(start, i-1):
    #     if pre_data['水流量'][j] > 0:
    #         a.append((pre_data['发生时间'][j+1]-pre_data['发生时间'][j-1]) / 2)

    last_use_index = data_index[data_index.index(i) - 1]
    use=pd.Timedelta(minutes=0)
    total_use=pd.Timedelta(minutes=0)
    hot_water=0
    used_quantity=0
    stop_times=0
    stop_time=pd.Timedelta(minutes=0)
    uses_cur=[]
    used_quantities_cur=[]
    stops=[]
    pre_stop=-1
    for j in range(start, last_use_index + 1):
        gap = get_gap(j)
        if pre_data['水流量'][j] > 0:
            use+=gap
            flow=data['水流量'][j]*60/gap.seconds
            used_quantity+=flow
            hot_water+=(data['热水量'][j]*flow)
            uses_cur.append(gap)
            used_quantities_cur.append(data['水流量'][j])
            if pre_stop>=0:
                stops.append(sum(list(map(lambda a:get_gap(a).seconds/60,range(pre_stop,j) )) ))
            pre_stop=-1
        else:
            # stop_time+=get_gap(j)
            if pre_data['水流量'][j-1]>0:
                stop_times+=1
                pre_stop=j

        total_use+=gap

    uses.append(use)
    total_uses.append(total_use)
    used_quantities.append(used_quantity)
    hot_waters.append(hot_water)
    stop_counts.append(stop_times)
    if stops:
        stop_time_waves.append(pd.Timedelta(minutes=np.array(stops).std()**2))
    else:
        stop_time_waves.append(pd.Timedelta(minutes=0))
    avg_flow=used_quantity*60/use.seconds
    flow_waves.append(sum(list(map(lambda a:((a[0]-avg_flow)**2)*a[1].seconds/60,zip(*(used_quantities_cur,uses_cur)) ) ) )*60/use.seconds)

    # stop_distances.append(stop_time)
    # if stop_times>0:
    #     stop_avgs.append(stop_time/stop_times)
    # else:
    #     stop_avgs.append(pd.Timedelta(minutes=0))
    pre_index = i
    indexs.append((start, last_use_index))
 # if data_index[data_index.index(i) - 1] == pre_index:
    #     total_uses.append(use)
    # else:
    # total_uses.append(pre_data['发生时间'][i] - pre_data['发生时间'][start] - pd.Timedelta(exceeds[i]))


deltas=np.array(total_uses)-np.array(uses)
use_rates=list(map(lambda a:(a[0].seconds/a[1].seconds),zip(*(uses,total_uses))))
stop_avgs=list(map(lambda a:pd.Timedelta(minutes=0) if a[1]==0 else a[0]/a[1],zip(*(deltas,stop_counts))))
stats=pd.DataFrame(np.array([pre_data['发生时间'][np.array(indexs)[:,0]],
                             total_uses,uses,stop_counts,deltas,stop_avgs,use_rates,
                             used_quantities,list(map(lambda a:a[0]*60/a[1].seconds,zip(*(used_quantities,uses)))),hot_waters,indexs,
                             flow_waves, stop_time_waves
                             ]).T
                   ,columns=['开始时间','总用时','用时','停顿次数','停顿时长','平均停顿时长','用水时长/总时长','用水量','平均水流量','热水量','用水下标','水流波动','停顿时间波动']) #

# stats=pd.DataFrame(np.array([pre_data['发生时间'][np.array(indexs)[:,0]],total_uses,uses,np.array(total_uses)-np.array(uses),used_quantities,hot_waters,indexs]).T,columns=['开始时间','总用时','用时','时差','用水量','热水量','用水下标']) #


for i in ['总用时','用时','停顿时长','平均停顿时长','停顿时间波动']:
    stats[i]=stats[i].apply(lambda a:("%d:%d:%d" % (a.seconds/3600,a.seconds%3600/60,a.seconds%60)))
for i in ['用水时长/总时长','用水量','平均水流量','热水量','水流波动']:
    stats[i] = stats[i].apply(lambda a:"%.2f" % a)
from IPython.display import display
print(stats)
display(stats)
# writer = pd.ExcelWriter('../tmp/water_heater2.xls', datetime_format='hh:mm:ss')
# stats.to_excel(writer)
stats.to_excel('../tmp/water_heater1.xls')


# x=np.arange(1,9,0.25)[4:]
# y=rates[3:]
# plt.plot(x,y,marker='o')
# plt.xticks(x)
# plt.yticks(y)
# plt.grid(True)
# # for a,b in zip(x,y):
# #     plt.text(a,b,(a,b),ha='center',va='bottom',withdash=True)
# plt.show()

# closes=[]
# pre_status=1
# close_time=0
# for i in data.index:
#     if data['水流量'][i]!=pre_status:
#         if pre_status==0:
#             closes.append((data['发生时间'][i]-close_time).seconds)
#         else:
#             close_time=data['发生时间'][i]
#         pre_status=data['水流量'][i]
# print(np.sum(np.array(closes)>60*4))


# print([i  if data['水流量'][i]==0])
# print(np.all((data['保温中']==1).__and__(data['加热剩余时间']>0)))
# print(data['开关机状态']=='开')