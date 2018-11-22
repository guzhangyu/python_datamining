import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF

class ArimaModel:

    def __init__(self,data):
        self.data=data

    # 寻找差分的阶数
    # data 要求key为时间，value为值的series
    def find_diff(self):
        # 平稳性检测
        diff = 0

        after_data = self.data
        adf = ADF(after_data)
        while adf[1] > 0.05 or not self.white_noise_test(after_data,('%d 阶差分' % diff)):
            print("adf:%f %s ------- %d" % (adf[1], adf[1] <= 0.05, diff))

            # if diff == len(data) - 1:
            #     raise Exception("diff:%d,len of data:%d" % (diff, len(data)))

            diff = diff + 1
            after_data = self.data.diff(diff).dropna()

            adf = ADF(after_data)

        print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))
        # print(adf)
        self.diff=diff
        return diff

    # 定阶
    def find_pq(self):
        pmax = int(len(self.data) / 10)  # 一般阶数不超过length/10
        qmax = int(len(self.data) / 10)  # 一般阶数不超过length/10
        bic_matrix = []  # bic矩阵
        for p in range(pmax + 1):
            tmp = []
            for q in range(qmax + 1):
                try:  # 存在部分报错，所以用try来跳过报错。
                    tmp.append(ARIMA(self.data, (p, self.diff, q)).fit(disp=-1).bic)
                except Exception as e:
                    # print("*** print sys.exc_info:")
                    # print(traceback.format_exc())
                    tmp.append(None)
            bic_matrix.append(tmp)

        bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

        p, q = bic_matrix.stack().astype(float).idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
        print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
        return p, q

    def gen_model(self):
        self.model = ARIMA(self.data, (self.p, self.diff, self.q)).fit()  # 建立并训练模型
        if self.diff>0:
            xdata_pred = self.model.predict(typ='levels')  # 预测
        else:
            xdata_pred = self.model.predict()
        # print("xdata_pred:")
        # print(xdata_pred)
        pred_error = (xdata_pred - self.data).dropna()  # 计算残差
        # print(pred_error)
        if not self.white_noise_test(pred_error,('ARIMA(%d,1,%d)' % (self.p,self.q))):
            raise Exception('模型白噪声检验未通过')
        return self.model

    def white_noise_test(self,data,s):
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb, pv = acorr_ljungbox(data)
        h = (pv < 0.05).sum()  # p值小于0.05，认为是非白噪声。
        print(("h.shape:%s" % pv.shape))
        if h > pv.shape[0]*0.05:
            print('%s非白噪声,p:%s' % (s,pv))
            return False
        else:
            print('%s是白噪声,p:%s' % (s, pv))
            return True

    def fit(self):
        self.find_diff()
        self.find_pq()
        self.gen_model()

    def predict(self,start, end):
        if self.diff>0:
            return self.model.predict(start=start, end=end, typ='levels')
        else:
            return self.model.predict(start=start, end=end)


    def calc_err(self,pre,predicted):
        rr = predicted - pre
        print(rr.mean(), ((rr ** 2).mean()) ** 0.5, (rr / np.abs(pre)).mean())


if __name__ == '__main__':
    start=pd.Timestamp('2018-11-01')
    print(start)
    a=[start+pd.Timedelta(days=i) for i in range(0,30)]
    data=pd.Series(index=a,data=np.random.random_sample(30))
    print(data)
    # print(data.diff(0).dropna())
    am=ArimaModel(data)
    am.fit()

    start=pd.Timestamp('2018-11-11')
    end=pd.Timestamp('2018-11-15')
    v=np.random.random_sample(5)
    r=am.predict(start,end)
    am.calc_err(v,r)