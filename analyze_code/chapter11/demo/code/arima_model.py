import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF

#data 要求key为时间，value为值的series
def check(data):
    after_data=data

    # 平稳性检测
    diff = 0
    adf = ADF(data)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    [[lb], [p]] = acorr_ljungbox(after_data, lags=1)
    while adf[1] > 0.05 or p<0.05:
        print("%f %s,%f %s ------- %d" % (adf[1],adf[1] <= 0.05,p,p>=0.05,diff))
        if diff == len(data)-1:
            raise Exception("diff:%d,len of data:%d" % (diff,len(data)))
        diff = diff + 1
        after_data=data.diff(diff).dropna()
        adf = ADF(after_data)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        [[lb], [p]] = acorr_ljungbox(after_data, lags=1)

    print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))
    print(adf)

    if p < 0.05:
        print('非白噪声,p:%f' % p)
    else:
        print('白噪声,p:%f' % p)
    return diff

def find_pq(xdata):
    # 定阶
    pmax = int(len(xdata) / 10)  # 一般阶数不超过length/10
    qmax = int(len(xdata) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # bic矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:  # 存在部分报错，所以用try来跳过报错。
                tmp.append(ARIMA(xdata, (p, 1, q)).fit().bic)
            except Exception as e:
                # print("*** print sys.exc_info:")
                # print(traceback.format_exc())
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

    p, q = bic_matrix.stack().astype(float).idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    return p,q

def model_check(xdata,p,q):
    arima = ARIMA(xdata, (p, 1, q)).fit()  # 建立并训练模型
    xdata_pred = arima.predict(typ='levels')  # 预测
    pred_error = (xdata_pred - xdata).dropna()  # 计算残差

    from statsmodels.stats.diagnostic import acorr_ljungbox
    [[lb], [pv]] = acorr_ljungbox(pred_error, lags=1)
    if pv < 0.05:
        print('ARIMA(%d,1,%d)非白噪声,p:%f' % (p,q,pv) )
    else:
        print('ARIMA(%d,1,%d)是白噪声,p:%f' % (p,q,pv))
    return arima

def model_predict(arima,start,end):
    return arima.predict(start=start,end=end,typ='levels')


start=pd.Timestamp('2018-11-01')
print(start)
a=[start+pd.Timedelta(days=i) for i in range(0,10)]
data=pd.Series(index=a,data=np.random.random_sample(10))
print(data)
# print(data.diff(0).dropna())
check(data)
p,q=find_pq(data)
model=model_check(data,p,q)

start=pd.Timestamp('2018-11-11')
end=pd.Timestamp('2018-11-15')
v=np.random.random_sample(5)
r=model_predict(model,start,end)
print(r)
rr=r-v
print(rr.mean(),((rr**2).mean())**0.5,(rr/np.abs(v)).mean())