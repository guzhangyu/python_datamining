#-*- coding: utf-8 -*-
#确定最佳p、d、q值
import pandas as pd
import sys
import traceback

#参数初始化
discfile = '../data/discdata_processed.xls'

data = pd.read_excel(discfile, index_col = 'COLLECTTIME')
pre_data=data
data = data.iloc[: len(data)-5] #不使用最后5个数据
xdata = data['CWXT_DB:184:D:\\']

from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(xdata, (0,1,1)).fit()
predicted=model.predict(typ='levels')
err=(predicted-xdata).dropna().round()


from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
lb, p= acorr_ljungbox(err, lags = 12)
print(p)
print(sum(p<0.05))


r=model.predict(start=pre_data.index[-5],end=pre_data.index[-1])
print(r)
print(pre_data.iloc[-5:]['CWXT_DB:184:D:\\'])
print(r-pre_data.iloc[-5:]['CWXT_DB:184:D:\\'].values)

# err_df=err.abs()
# print(err_df.mean(),(err_df**2).mean()**0.5,(err_df/xdata).std())

# r=model.predict(data.iloc[-5:]['CWXT_DB:184:D:\\'])
# print(r)
