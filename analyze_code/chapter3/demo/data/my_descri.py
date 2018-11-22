import pandas as pd

data=pd.read_csv('/Users/zhangyugu/Desktop/af_data.csv',index_col=u'日期',parse_dates=[0],delimiter=',',dtype=object,encoding='gbk',usecols=range(0,9))

print(data.describe())

print(data)
