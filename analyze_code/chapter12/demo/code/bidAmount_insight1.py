import pandas as pd
import json
import datetime
import numpy as np

import sys

sys.setrecursionlimit(1000000)


import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def str2Date(strvalue):
    if not strvalue or type(strvalue) == float or '0001' == strvalue[:4]:
        return datetime.date.max
    # print(type(str))
    # print(type(str) == float)
    strvalue=strvalue[:10]
    # print(str)
    return datetime.datetime.strptime(strvalue,"%Y-%m-%d").date()

# print(str2Date('2018-11-08 14:51:40'))
# exit(0)

rpt_fb_adactivitys=pd.read_csv('~/Downloads/rpt_fb_adactivity (2).csv')
rpt_fb_adactivitys['old_value']=rpt_fb_adactivitys['extra_data'].apply(lambda a:json.loads(a)['additional_value']['old_value'] )
rpt_fb_adactivitys['new_value']=rpt_fb_adactivitys['extra_data'].apply(lambda a:json.loads(a)['additional_value']['new_value'] )
rpt_fb_adactivitys['insight_date']=rpt_fb_adactivitys['date(event_time)']

rpt_insight_alls=pd.read_csv('~/Downloads/fa (2).csv')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 6

rpt_insight_alls=rpt_insight_alls.sort_values(by=['fb_application_id','country','fb_adset_id','insight_date'])
fb_application_id,country,fb_adset_id=None,None,None
insight_datess,installss,spendss,cpiss,fb_adset_ids,fb_create_times,arichived_time = [],[],[],[],[],[],[]
insight_dates,installs,spends,cpis= [],[],[],[]
installss.append(installs)
insight_datess.append(insight_dates)
spendss.append(spends)
cpiss.append(cpis)

def plot_axs1(datass,name):
    # print(datass)
    new_datas2,new_fb_adset_ids,max_insight_datess1,fb_create_times1=split_data(datass)
    for i in range(0,len(new_datas2)):
        plot_axs(new_datas2[i],name,new_fb_adset_ids[i],max_insight_datess1[i],fb_create_times1[i],i)

def plot_axs(datassi,name,new_fb_adset_ids,max_insight_datess1,fb_create_timess,indexNo):
    fig = plt.figure(figsize=(12, 9))
    l = "%s-%s" % (fb_application_id, country)
    fig.suptitle(l + ' '+name)
    ax1 = fig.gca()
    for adsetIdIndex in range(0, len(datassi)):
        max_insight_dates=max_insight_datess1[adsetIdIndex]
        fb_adset_id = new_fb_adset_ids[adsetIdIndex]
        color = randomcolor()
        # print(max_insight_dates, datass[adsetIdIndex])
        ax1.plot_date(max_insight_dates, datassi[adsetIdIndex], color=color, ls='-') #, label=str(fb_adset_id)
        if sum(rpt_fb_adactivitys['object_id'] == fb_adset_id) > 0:
            datas = rpt_fb_adactivitys[rpt_fb_adactivitys['object_id'] == fb_adset_id]
            # print("dates: %s" % dates)
            # print(datas)
            for k in datas.index:
                data = datas.T[k]
                date = str2Date(data['date(event_time)'])
                if date==fb_create_timess[adsetIdIndex]:
                    continue
                strv = '%s -> %s' % (data['old_value'], data['new_value'])
                ax1.axvline(date, color=color, ls='-')
                cur_dates=[a.date() for a in max_insight_dates]
                if date in cur_dates:
                    y=datassi[adsetIdIndex][cur_dates.index(date)]
                else:
                    y=max(datassi[adsetIdIndex])
                    print("warn! %s not in %s" %(date,cur_dates))
                ax1.text(x=date, y=y*1.1, s=strv,color=color)

    ax1.legend(loc='best')
    fig.savefig(fname='/Users/zhangyugu/python_works/tmp/images1/%s-%d-%s.png' % (l,indexNo,name))

def split_data(datass):
    maxs=[]
    for datas1 in datass:
        maxs.append(max(datas1))
    # print(datass)
    # print(maxs)
    max1=max(maxs)
    levels=[max1]
    while max1>10:
        max1=max1/10
        levels.append(max1)
    levels=sorted(levels)

    new_datas,new_fb_adset_ids_after,max_insight_datess1,fb_create_times1 = [],[],[],[]
    for level in levels:
        new_datas.append([])
        new_fb_adset_ids_after.append([])
        max_insight_datess1.append([])
        fb_create_times1.append([])

    j=0
    for datas1 in datass:
        i=0
        for level in levels:
            if max(datas1) <=level:
                new_fb_adset_ids_after[i].append(fb_adset_ids[j])
                max_insight_datess1[i].append(max_insight_datess[j])
                fb_create_times1[i].append(fb_create_times[j])
                new_datas[i].append(datas1)
                break
            i = i + 1
        j=j+1
    return new_datas,new_fb_adset_ids_after,max_insight_datess1,fb_create_times1


MIN_DATE=str2Date('2018-11-08')
MAX_DATE=str2Date('2018-11-23')
for i in rpt_insight_alls.index:
    a=rpt_insight_alls.T[i]
    if (fb_application_id,country,fb_adset_id) == (None,None,None):
        fb_application_id, country, fb_adset_id=a['fb_application_id'],a['country'],a['fb_adset_id']
        fb_adset_ids.append(fb_adset_id)
        fb_create_times.append(str2Date(a['fb_create_time']))
        arichived_time.append(str2Date(a['archived_time']))

    if (fb_application_id, country) != (a['fb_application_id'],a['country']):
        new_installss,new_spendss,new_cpiss,max_insight_datess=[],[],[],[]
        for adsetIdIndex in range(0, len(insight_datess)):
            insight_dates=insight_datess[adsetIdIndex]
            installs=installss[adsetIdIndex]
            spends=spendss[adsetIdIndex]
            cpis=cpiss[adsetIdIndex]

            min_insight_date = max(MIN_DATE ,fb_create_times[adsetIdIndex])
            max_insight_date = min(MAX_DATE,arichived_time[adsetIdIndex])
            new_installs,new_spends,new_cpis=[],[],[]
            new_installss.append(new_installs)
            new_spendss.append(new_spends)
            new_cpiss.append(new_spends)
            max_insight_dates = pd.date_range(min_insight_date, max_insight_date)
            max_insight_datess.append(max_insight_dates)
            for dt in max_insight_dates:
                dt=dt.date()
                if dt in insight_dates:
                    index=insight_dates.index(dt)
                    new_installs.append(installs[index])
                    new_spends.append(spends[index])
                    new_cpis.append(cpis[index])
                else:
                    new_installs.append(0)
                    new_spends.append(0)
                    new_cpis.append(None)

        plot_axs1(new_installss,' installs ')
        plot_axs1(new_spendss, ' spends ')
        plot_axs1(new_cpiss, ' cpis ')
        insight_datess, installss, spendss, cpiss, fb_adset_ids,fb_create_times,arichived_time  = [], [], [], [], [],[],[]

    if fb_adset_id != a['fb_adset_id']:
        fb_adset_ids.append(a['fb_adset_id'])
        fb_create_times.append(str2Date(a['fb_create_time']))
        arichived_time.append(str2Date(a['archived_time']))

        insight_dates, installs, spends, cpis = [], [], [], []
        installss.append(installs)
        insight_datess.append(insight_dates)
        spendss.append(spends)
        cpiss.append(cpis)

    fb_application_id, country, fb_adset_id = a['fb_application_id'], a['country'], a['fb_adset_id']

    insight_dates.append(str2Date(a['insight_date']))
    # d=d.sum()
    # print(d)
    installs.append(a['sum(installs_num)'])
    spends.append(a['sum(spend)'])
    if a['sum(installs_num)'] and a['sum(installs_num)'] > 0:
        cpis.append(a['sum(spend)'] / a['sum(installs_num)'])
    else:
        cpis.append(0)

# plt.legend(loc='best')
# plt.savefig()
# plt.show()