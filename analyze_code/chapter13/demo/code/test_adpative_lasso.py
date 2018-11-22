import numpy as np
from glmnet_py import glmnet
import glmnet_py as gp
import pandas as pd
from numpy import Inf
from sklearn.linear_model.base import LinearRegression

# rng = np.random.RandomState(0)
# y = rng.randn(6)
# X = rng.randn(6, 5)
inputfile = '../data/data1.csv' #输入的数据文件
data = pd.read_csv(inputfile) #读取数据
X=np.array(data.iloc[:,:-1])
y=np.array(data.iloc[:,-1])
# w = 1.0 + rng.rand(6)
# print(y,X,w)
#
# clf = LinearRegression()
# clf.fit(X, y, sample_weight=w)
# coefs1 = clf.coef_
#
# scaled_y = y * np.sqrt(w)
# scaled_X = X * np.sqrt(w)[:, np.newaxis]
# print(scaled_X)
# clf.fit(scaled_X, scaled_y)
# coefs2 = clf.coef_
#
# print(coefs1)
#
# print(coefs2)


# print(X)
f=glmnet(x=X,y=y,family='gaussian',alpha=0,standardize=True) #parallel=True,
xc=X.shape[1]

d=gp.glmnetCoef(f,s=np.array([min(f['lambdau'])]))
w3=1/np.abs(d[1:(xc+1)])
f=glmnet(x=X,y=y,family='gaussian',alpha=1,standardize=True,penalty_factor=w3) #parallel=True,
# print(sum(w3==Inf))
print(gp.glmnetCoef(f,s=np.array([min(f['lambdau'])])))