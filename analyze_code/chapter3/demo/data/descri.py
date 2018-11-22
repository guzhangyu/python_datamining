import pandas as pd
import numpy as np

catering_sale = 'catering_sale.xls'
data=pd.read_excel(catering_sale,index_col=u'日期')
stat=data.describe()
print(stat)
sale_stat=stat['销量']


sale=data['销量'].values
xx=data[sale>0]
pre_xx=xx

#箱线图法
half= (sale_stat['75%'] - sale_stat['25%']) / 2
xx=xx[xx['销量'].values<=sale_stat['75%']+half]
xx=xx[xx['销量'].values>=(sale_stat['25%']-half)]


#3 原则
#xx=xx[(xx['销量'].values-sale_stat['mean'])<=3*sale_stat['std']]

print(len(xx))

#获取两个dataframe的差集
def sub(pre_xx,xx):
    pre1 = pre_xx.drop_duplicates()
    xx1 = xx.drop_duplicates()
    return pre1.append(xx1).drop_duplicates(keep=False)

import matplotlib.pyplot as plt
import matplotlib as mpl
cp_xx=sub(pre_xx,xx)
print(len(cp_xx))

mpl.rcParams['font.sans-serif'] = ['SimHei']
fig=plt.figure(4,figsize=(10,8))

ax=fig.add_subplot(412)
ax.plot(xx['销量'].index.values,xx['销量'].values)
ax.scatter(cp_xx['销量'].index.values,cp_xx['销量'].values,color='r')
# ax.boxplot(data['销量'].values.T)

ax=fig.add_subplot(411)
ax=data.boxplot()
# xx['销量'].plot()
# len=len(data['销量'].keys())
# data['日期']=np.linspace(0,len,len)
#     #data['销量'].keys()
# data.plot(kind='scatter',x='日期',y='销量')
# data1=pd.read_excel(catering_sale)
# data1.plot(kind='scatter',x='日期',y='销量')

ax=fig.add_subplot(413)
ax.plot(xx['销量'].index.values,xx['销量'].values)
stat=xx.describe()
print(stat)

#频率分布
xl=np.linspace(stat['销量']['min']-1,stat['销量']['max'],7)
f=[]
pre=0
for x in xl:
    pre1=(xx['销量'].values<=x).sum()
    f.append(pre1-pre)
    pre=pre1
for i in range(0,len(f)):
    f[i]=f[i]/sum(f)


ax=fig.add_subplot(414)
xx['销量'].hist(histtype='stepfilled',bins=np.linspace(stat['销量']['min']-1,stat['销量']['max'],7),stacked=True)#.index.values,bins=6
ax.grid(True)


# ax.set_xticks(np.linspace(stat['销量']['min']-1,stat['销量']['max'],7))
# ax=fig.add_subplot(212)
# ax.
plt.show()
# print(len(pre_xx[set(pre_xx['日期'])-set(xx['日期'])]))

