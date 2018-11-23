import pandas as pd
import json
import datetime
import numpy as np


import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def toDate(str):
    if not str or type(str) == float or '0001' == str[:4]:
        return datetime.date.max
    # print(type(str))
    # print(type(str) == float)
    str=str[:10]
    # print(str)
    return datetime.datetime.strptime(str,"%Y-%m-%d").date()

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
MIN_DATE=toDate('2018-11-08')
MAX_DATE=toDate('2018-11-23')

def plot_axs(datass,name):
    fig = plt.figure(figsize=(12, 9))
    l = "%s - %s" % (fb_application_id, country)
    fig.suptitle(l + ' '+name)
    ax1 = fig.gca()
    for adsetIdIndex in range(0, len(datass)):
        max_insight_dates=max_insight_datess[adsetIdIndex]
        fb_adset_id = new_fb_adset_ids[adsetIdIndex]
        color = randomcolor()
        ax1.plot_date(max_insight_dates, datass[adsetIdIndex], color=color, ls='-', label=str(fb_adset_id))
        if sum(rpt_fb_adactivitys['object_id'] == fb_adset_id) > 0:
            datas = rpt_fb_adactivitys[rpt_fb_adactivitys['object_id'] == fb_adset_id]
            # print("dates: %s" % dates)
            # print(datas)
            for k in datas.index:
                data = datas.T[k]
                date = data['date(event_time)']
                strv = '%s -> %s' % (data['old_value'], data['new_value'])
                ax1.axvline(date, color=color, ls='-')
                ax1.text(x=date, y=max(datass[adsetIdIndex]) / 2, s=strv)

    ax1.legend()


for i in rpt_insight_alls.index:
    a=rpt_insight_alls.T[i]
    if (fb_application_id,country,fb_adset_id) == (None,None,None):
        fb_application_id, country, fb_adset_id=a['fb_application_id'],a['country'],a['fb_adset_id']
        fb_adset_ids.append(fb_adset_id)

    if (fb_application_id, country) != (a['fb_application_id'],a['country']):
        new_installss,new_spendss,new_cpiss,max_insight_datess,new_fb_adset_ids=[],[],[],[],fb_adset_ids
        for adsetIdIndex in range(0, len(insight_datess)):
            insight_dates=insight_datess[adsetIdIndex]
            installs=installss[adsetIdIndex]
            spends=spendss[adsetIdIndex]
            cpis=cpiss[adsetIdIndex]

            min_insight_date = max(MIN_DATE , toDate(fb_create_times[adsetIdIndex]))
            max_insight_date = min(MAX_DATE, toDate(arichived_time[adsetIdIndex]))
            new_installs,new_spends,new_cpis=[],[],[]
            new_installss.append(new_installs)
            new_spendss.append(new_spends)
            new_cpiss.append(new_spends)
            max_insight_dates = pd.date_range(min_insight_date, max_insight_date)
            max_insight_datess.append(max_insight_dates)
            for dt in max_insight_dates:
                if dt in insight_dates:
                    index=insight_dates.index(dt)
                    new_installs.append(installs[index])
                    new_spends.append(spends[index])
                    new_cpis.append(cpis[index])
                else:
                    new_installs.append(0)
                    new_spends.append(0)
                    new_cpis.append(None)

        plot_axs(new_installs,' installs ')
        plot_axs(new_spends, ' spends ')
        plot_axs(new_cpis, ' cpis ')
        insight_datess, installss, spendss, cpiss, fb_adset_ids,fb_create_times,arichived_time  = [], [], [], [], [],[],[]

    if fb_adset_id != a['fb_adset_id']:
        fb_adset_ids.append(fb_adset_id)
        fb_create_times.append(a['fb_create_time'])
        arichived_time.append(a['archived_time'])

        insight_dates, installs, spends, cpis = [], [], [], []
        insight_datess.append(insight_dates)
        installss.append(installs)
        spendss.append(spends)
        cpiss.append(cpis)

    fb_application_id, country, fb_adset_id = a['fb_application_id'], a['country'], a['fb_adset_id']

    insight_dates.append(a['insight_date'])
    # d=d.sum()
    # print(d)
    installs.append(a['sum(installs_num)'])
    spends.append(a['sum(spend)'])
    if a['sum(installs_num)'] and a['sum(installs_num)'] > 0:
        cpis.append(a['sum(spend)'] / a['sum(installs_num)'])
    else:
        cpis.append(0)

plt.legend(loc='best')
plt.show()