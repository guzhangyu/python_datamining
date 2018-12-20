import numpy as np

def GM11(x0): #自定义灰色预测函数
  x1 = x0.cumsum() #1-AGO序列
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std()
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率

class LassoModel:
    def __init__(self,data):
        self.data=data

    def glmnet(self,data,test=None):
        from glmnet_py import glmnet
        import glmnet_py as gp
        X = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])
        #一维连续因变量
        f = glmnet(x=X, y=y, family='gaussian', alpha=0, standardize=True)  # parallel=True,

        #lambda 是惩罚系数 alpha=0 表示线性
        s = self.getGlmnetS(f)

        d = gp.glmnetCoef(f, s=s)
        w3 = 1 / np.abs(d[1:(X.shape[1] + 1)]) #惩罚系数
        f = glmnet(x=X, y=y, family='gaussian', alpha=1, standardize=True, penalty_factor=w3)  # parallel=True,
        # print(sum(w3==Inf))

        s=self.getGlmnetS(f)
        coef_=gp.glmnetCoef(f,s=s)
        # coef_=coef_[:,-1]
        coef_=np.reshape(coef_,(1,-1))[0]
        # print(coef_)
        self.columns = data.columns[coef_ != 0][:-1] #抛弃临界的最后一个

        if test is not None:
            return gp.glmnetPredict(f,newx=np.array(test.iloc[:, :-1]),s=s,ptype='response')

    def getGlmnetS(self,f):
        v = self.maxInStd(f['lambdau'])
        if np.isnan(v):
            v = 0
        return np.array([v])

    def maxInStd(self,a):
        # idx= a>=(a.mean()-a.std())
        # if sum(idx)>0:
        #     return sorted(a[idx])[-1]
        # return sorted(a)[0]
        return min(a)

    def fit(self,data):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
        model.fit(data.iloc[:, 0:-1],data.iloc[:,-1])
        # print(model.coef_>0)  # 各个特征的系数
        self.columns=data.columns[:-1][model.coef_!=0]
        # print(self.columns)

    #灰度，也是一种预测
    def gray(self,data):
        # data.loc[-2:]=None
        for i in self.columns:
            f = GM11(data[i][:-2].values)[0]
            data[i][-2]=f(len(data)-1)
            data[i][-1] = f(len(data))
            data[i] = data[i].round(2)

    def predict(self,data):
        data_train=data.loc[:len(data)-3]
        data_mean = data_train.mean()
        data_std = data_train.std()
        data_train = (data_train - data_mean) / data_std  # 数据标准化
        x_train = data_train[self.columns].as_matrix()  # 特征数据
        y_train = data_train.iloc[:,-1].as_matrix()  # 标签数据

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation

        feature=self.columns
        n=len(self.columns)
        model = Sequential()  # 建立模型
        model.add(Dense(input_dim=n, units=n*2))
        model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
        model.add(Dense(input_dim=n*2, units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型

        model.fit(x_train, y_train, nb_epoch=1500, batch_size=16)  # 训练模型，学习一万五千次
        # model.save_weights(modelfile)  # 保存模型参数

        # 预测，并还原结果。
        x = ((data[feature] - data_mean[feature]) / data_std[feature]).as_matrix()
        data[u'y_pred'] = model.predict(x) * data_std[-1] + data_mean[-1]
        data[u'y_pred'] = data[u'y_pred'].round()

        data['y']=data.iloc[:,-1]
        import matplotlib.pyplot as plt  # 画出预测结果图
        data[['y', 'y_pred']].plot(style=['b-o', 'r-*'])
        plt.show()

    def deep_learn_train(self,data_train,test=None):
        # data_mean = data_train.mean()
        # data_std = data_train.std()
        # data_train = (data_train - data_mean) / data_std  # 数据标准化
        x_train = data_train[self.columns].values  # 特征数据
        y_train = data_train.iloc[:, -1].values  # 标签数据

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation

        n = len(self.columns)
        model = Sequential()  # 建立模型
        model.add(Dense(input_dim=n, units=n * 4))
        model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
        model.add(Dense(input_dim=n * 4, units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型

        model.fit(x_train, y_train, nb_epoch=1500, batch_size=32)  # 训练模型，学习一万五千次
        # model.save_weights(modelfile)  # 保存模型参数
        # if test is not None:
        #     data_mean = test.mean()
        #     data_std = test.std()
        #     # 预测，并还原结果。
        #     feature = self.columns
        #     x = ((test[feature] - data_mean[feature]) / data_std[feature]).as_matrix()
        #     print(x)
        #     r = model.predict(x)
        #     print(r)
        self.model=model
        # self.mean=data_mean[-1]
        # self.std=data_std[-1]

    def deep_learn_train1(self,data_train,test=None):
        # data_mean = data_train.mean()
        # data_std = data_train.std()
        # data_train = (data_train - data_mean) / data_std  # 数据标准化
        x_train = data_train[self.columns].as_matrix()  # 特征数据
        y_train = data_train.iloc[:, -1].as_matrix()  # 标签数据

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation

        n = len(self.columns)
        model = Sequential()  # 建立模型
        model.add(Dense(input_dim=n, units=n * 2))
        model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
        model.add(Dense(input_dim=n * 2, units=1))
        model.add(Activation('sigmoid'))  # 由于是0-1输出，用sigmoid函数作为激活函数

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # class_mode = 'binary'

        model.fit(x_train, y_train, nb_epoch=1500, batch_size=16)  # 训练模型，学习一万五千次
        if test:
            print(model.predict(test))
        # model.save_weights(modelfile)  # 保存模型参数
        self.model=model
        # self.mean=data_mean[-1]
        # self.std=data_std[-1]

    def lr(self, x_train, y_train, x_test):
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.linear_model import RandomizedLogisticRegression as RLR
        rlr = RLR()  # 建立随机逻辑回归模型，筛选变量
        rlr.fit(x_train, y_train)  # 训练模型
        rlr.get_support()  # 获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
        print(u'通过随机逻辑回归模型筛选特征结束。')
        print(u'有效特征为：%s' % ','.join(x_train.columns[rlr.get_support()]))
        print(rlr.get_support())
        # x = data[data.columns[:-1][rlr.get_support()]].values #筛选好特征
        print('rlr : %s' % rlr.all_scores_)

        lr = LR()  # 建立逻辑回归模型
        lr.fit(x_train, y_train)  # 用筛选后的特征数据来训练模型
        print(u'逻辑回归模型训练结束。')
        print(u'模型的平均正确率为：%s' % lr.score(x_train, y_train))  # 给出模型的平均正确率，本例为81.4%
        y_test1 = lr.predict(x_test)
        return y_test1

    def predict_data(self,data):
        # data_mean = data.mean()
        # data_std = data.std()
        # # 预测，并还原结果。
        # feature = self.columns
        # x = ((data[feature] - data_mean[feature]) / data_std[feature]).as_matrix()
        # r = self.model.predict(x) * self.std + self.mean
        # print(r)
        # r = r.round()
        # return r
        x_test=data[self.columns]
        y_test=data.iloc[:,-1]
        score = self.model.evaluate(x_test, y_test, batch_size=16)
        print(score)
        return self.model.predict(x_test)

    def adaptive_lasso(self,pre_data):
        import numpy as np
        from sklearn.linear_model import Lasso
        data=pre_data.copy()
        X=data.iloc[:, :-1]
        y=data.iloc[:, -1]
        X /= np.sum(X ** 2, axis=0)  # scale features

        alpha = 0.1

        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

        # Or another option:
        # ll = 0.01
        # g = lambda w: np.log(ll + np.abs(w))
        # gprime = lambda w: 1. / (ll + np.abs(w))

        n_samples, n_features = X.shape
        p_obj = lambda w: 1. / (2 * n_samples) * np.sum((y - np.dot(X, w)) ** 2) \
                          + alpha * np.sum(g(w))

        weights = np.ones(n_features)
        n_lasso_iterations = 5

        for k in range(n_lasso_iterations):
            # print(X)
            # print(np.repeat(weights[np.newaxis, :],n_samples,axis=0))
            X_w = X / np.repeat(weights[np.newaxis, :],n_samples,axis=0) #weights[np.newaxis, :]
            clf = Lasso(alpha=alpha, fit_intercept=False)
            # print(X_w,y)
            clf.fit(X_w, y)
            coef_ = clf.coef_ / weights
            weights = gprime(coef_)
            print(coef_)
            print(p_obj(coef_))  # should go down


if __name__ == '__main__':
    import pandas as pd

    inputfile = '../data/data1.csv'  # 输入的数据文件
    data = pd.read_csv(inputfile)  # 读取数据
    model = LassoModel(data)
    model.glmnet(data)
    # model.fit(data)
    model.gray(data)
    model.predict(data)