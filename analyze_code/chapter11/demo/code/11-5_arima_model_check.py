#-*- coding: utf-8 -*-
#模型检验
import pandas as pd
import numpy as np

#参数初始化
discfile = '../data/discdata_processed.xls'
lagnum = 12 #残差延迟个数

data = pd.read_excel(discfile, index_col = 'COLLECTTIME')
pre_data=data
data = data.iloc[: len(data)-5] #不使用最后5个数据
xdata = data['CWXT_DB:184:D:\\']

from statsmodels.tsa.arima_model import ARIMA #建立ARIMA(0,1,1)模型

arima = ARIMA(xdata, (0, 1, 1)).fit() #建立并训练模型
xdata_pred = arima.predict(typ = 'levels') #预测
pred_error = (xdata_pred - xdata).dropna() #计算残差

from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验

lb, p= acorr_ljungbox(pred_error, lags = lagnum)
print(p)
h = (p < 0.05).sum() #p值小于0.05，认为是非白噪声。
if h > 0:
  print(u'模型ARIMA(0,1,1)不符合白噪声检验')
else:
  print(u'模型ARIMA(0,1,1)符合白噪声检验')

r=(arima.predict(start=pre_data.index[-5],end=pre_data.index[-1],typ='levels')).values/(10**6)
print(r)
pre=pre_data.iloc[-5:]['CWXT_DB:184:D:\\'].values/(10**6)
print(pre)
rr=r-pre
rr=np.abs(rr)
print(rr)
print(rr.mean(),((rr**2).mean())**0.5,(rr/pre).mean())

xdata1 = pre_data['CWXT_DB:184:D:\\']
arima=ARIMA(xdata1,(0,1,1)).fit()
start=pre_data.index[-1]+pd.Timedelta(days=1)
print(pre_data.index[-5])
print(start)
print(start+pd.Timedelta(days=4))
r=arima.predict(start=start,end=start+pd.Timedelta(days=4),typ='levels')
r=r.values/157283328
print(r)