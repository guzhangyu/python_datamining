import pandas as pd
import numpy as np

data=pd.read_excel('拓展思考样本数据.xls')
dt={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7}
data['空气等级']=[dt[i] for i in data['空气等级']]
# print(np.unique(data['空气等级']))
columns=data.columns
data=data.values
# print(data)
np.random.shuffle(data)
# print(data)
# print(0.8*len(data))
train=data[:int(0.8*len(data)),:]
test=data[int(0.8*len(data)):,:]
# print(train)

from sklearn import svm
model=svm.SVC()
model.fit(train[:,:6],train[:,6])

import pickle
pickle.dump(model,open('tmp/svm.model','wb'))

from sklearn import metrics
predicted_test=model.predict(test[:,:6])
cm_test=metrics.confusion_matrix(test[:,6],predicted_test)
cm_train=metrics.confusion_matrix(train[:,6],model.predict(train[:,:6]))
print(predicted_test)
# print(np.unique(test[:,6]))
print(cm_test)
print(cm_train)
pd.DataFrame(cm_test).to_excel('tmp/test.xls')
pd.DataFrame(cm_train).to_excel('tmp/train.xls')