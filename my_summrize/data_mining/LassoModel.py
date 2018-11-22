from GM11 import GM11 #引入自己编写的灰色预测函数

class LassoModel:
    def __init__(self,data):
        self.data=data

    def glmnet(self,data):
        import numpy as np
        from glmnet_py import glmnet
        import glmnet_py as gp
        X = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])
        f = glmnet(x=X, y=y, family='gaussian', alpha=0, standardize=True)  # parallel=True,

        #lambda 是惩罚系数 alpha=0 表示线性
        d = gp.glmnetCoef(f, s=np.array([min(f['lambdau'])]))
        w3 = 1 / np.abs(d[1:( X.shape[1] + 1)])
        f = glmnet(x=X, y=y, family='gaussian', alpha=1, standardize=True, penalty_factor=w3)  # parallel=True,
        # print(sum(w3==Inf))
        coef_=gp.glmnetCoef(f, s=np.array([min(f['lambdau'])]))
        coef_=np.reshape(coef_,(1,-1))[0]
        # print(coef_)
        self.columns = data.columns[coef_ != 0][:-1]

    def fit(self,data):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
        model.fit(data.iloc[:, 0:-1],data.iloc[:,-1])
        # print(model.coef_>0)  # 各个特征的系数
        self.columns=data.columns[:-1][model.coef_!=0]
        # print(self.columns)

    def gray(self,data):
        # data.loc[-2:]=None
        for i in self.columns:
            f = GM11(data[i][:-2].values)[0]
            data[i][-2]=f(len(data)-1)
            data[i][-1] = f(len(data))
            data[i] = data[i].round(2)

    def predict(self,data):
        data_train=data.loc[:len(data)-3]
        # print(data_train)
        data_mean = data_train.mean()
        data_std = data_train.std()
        data_train = (data_train - data_mean) / data_std  # 数据标准化
        x_train = data_train[self.columns].as_matrix()  # 特征数据
        y_train = data_train['y'].as_matrix()  # 标签数据

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation

        feature=self.columns
        n=len(self.columns)
        model = Sequential()  # 建立模型
        model.add(Dense(input_dim=n, units=n*2))
        model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
        model.add(Dense(input_dim=n*2, units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型

        model.fit(x_train, y_train, nb_epoch=15000, batch_size=16)  # 训练模型，学习一万五千次
        # model.save_weights(modelfile)  # 保存模型参数

        # 预测，并还原结果。
        x = ((data[feature] - data_mean[feature]) / data_std[feature]).as_matrix()
        data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
        data[u'y_pred'] = data[u'y_pred'].round()

        import matplotlib.pyplot as plt  # 画出预测结果图
        # print(data[['y', 'y_pred']])
        p = data[['y', 'y_pred']].plot(style=['b-o', 'r-*'])
        plt.show()



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





import pandas as pd
inputfile = '../data/data1.csv' #输入的数据文件
data = pd.read_csv(inputfile) #读取数据
model=LassoModel(data)
model.glmnet(data)
# model.fit(data)
model.gray(data)
model.predict(data)