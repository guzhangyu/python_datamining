import pandas as pd
import json

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color



rpt_fb_adactivitys=pd.read_csv('~/Downloads/rpt_fb_adactivity (1).csv')
rpt_fb_adactivitys['old_value']=rpt_fb_adactivitys['extra_data'].apply(lambda a:json.loads(a)['additional_value']['old_value'] )
rpt_fb_adactivitys['new_value']=rpt_fb_adactivitys['extra_data'].apply(lambda a:json.loads(a)['additional_value']['new_value'] )
rpt_fb_adactivitys['fb_adset_id']=rpt_fb_adactivitys['object_id']
rpt_fb_adactivitys['insight_date']=rpt_fb_adactivitys['date(event_time)']
del rpt_fb_adactivitys['extra_data']

rpt_insight_alls=pd.read_csv('~/Downloads/fa.csv')
import matplotlib.pyplot as plt
insights_app_country=rpt_insight_alls.groupby(by=['fb_application_id','country'])
for ac,data in insights_app_country:

    fig=None
    # plt.axis.label.text("%s - %s" % (ac[0], ac[1]))
    i=1
    sums = data.groupby(by=['fb_adset_id'])
    for a,b in sums:
        # print(a,b.groupby(by='insight_date').sum())
        rr=b.groupby(by='insight_date')
        if len(rr.groups)<=1:
            continue
        if not fig:
            fig = plt.figure()
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
        insight_dates=[]
        installs=[]
        spends=[]
        cpis=[]
        for c,d in rr:
            insight_dates.append(c)
            # d=d.sum()
            # print(d)
            installs.append(d['sum(installs_num)'])
            spends.append(d['sum(spend)'])
            if d['sum(installs_num)'] and d['sum(installs_num)']>0:
                cpis.append(d['sum(spend)']/d['sum(installs_num)'])
            else:
                cpis.append(0)
        # print(spends)
        color = randomcolor()
        ax1.plot_date(insight_dates, installs, color=color, ls='-')
        ax2.plot_date(insight_dates, spends, color=color, ls='-')
        ax3.plot_date(insight_dates, cpis, color=color, ls='-')
        # if i==30:
        #     break
        i=i+1

plt.legend()
plt.show()
    #print(insight_dates)
    # plt.plot()

# result=pd.merge(rpt_fb_adactivitys,rpt_insight_alls,how='inner',on=['fb_adset_id','insight_date'])
#
# fb_adset_id_cur=None

# for a in result.iloc[:]:
#     if not fb_adset_id_cur or fb_adset_id_cur
#     plt.plot()
# r=result.groupby(by=['fb_adset_id','country'])
# print(r)
# result=result.sort_values(by=['fb_adset_id','country','date(event_time)'])



# result.groupby(by='fb_adset_id').count(level='insight_date')

# y=result.copy()[['fb_adset_id','insight_date']].drop_duplicates()
#
# x=y['fb_adset_id'].value_counts()
# print(x)
# idx=(x[x>1]).index
# print(idx)
# print(result['fb_adset_id'].count())
# print(sum(result['fb_adset_id'].isin(set(idx))))
# result=result[result['fb_adset_id'].isin(list(idx))]
# result=result.sort_values(by=['fb_adset_id','country','date(event_time)'])
# print(result)