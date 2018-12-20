import numpy as np
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
        s = self.__getGlmnetS(f)

        d = gp.glmnetCoef(f, s=s)
        w3 = 1 / np.abs(d[1:(X.shape[1] + 1)]) #惩罚系数
        f = glmnet(x=X, y=y, family='gaussian', alpha=1, standardize=True, penalty_factor=w3)  # parallel=True,
        # print(sum(w3==Inf))

        s=self.__getGlmnetS(f)
        coef_=gp.glmnetCoef(f,s=s)
        # coef_=coef_[:,-1]
        coef_=np.reshape(coef_,(1,-1))[0]
        # print(coef_)
        self.columns = data.columns[coef_ != 0][:-1] #抛弃临界的最后一个

        if test is not None:
            return gp.glmnetPredict(f,newx=np.array(test.iloc[:, :-1]),s=s,ptype='response')
        return self.columns

    def __getGlmnetS(self,f):
        v = self.__minInStd(f['lambdau'])
        if np.isnan(v):
            v = 0
        return np.array([v])

    def __minInStd(self,a):
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