#-*- coding: utf-8 -*-
import pandas as pd

inputfile = '../data/moment.csv' #�����ļ�
outputfile1 = '../tmp/cm_train.xls' #ѵ�������������󱣴�·��
outputfile2 = '../tmp/cm_test.xls' #���������������󱣴�·��
data = pd.read_csv(inputfile, encoding = 'gbk') #��ȡ���ݣ�ָ������Ϊgbk
data = data.as_matrix()

from numpy.random import shuffle #�����������
shuffle(data) #�����������
data_train = data[:int(0.8*len(data)), :] #ѡȡǰ80%Ϊѵ������
data_test = data[int(0.8*len(data)):, :] #ѡȡǰ20%Ϊ��������

#���������ͱ�ǩ
x_train = data_train[:, 2:]*30
y_train = data_train[:, 0].astype(int)
x_test = data_test[:, 2:]*30
y_test = data_test[:, 0].astype(int)

#����ģ����صĺ�������������ѵ��ģ��
from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
pickle.dump(model, open('../tmp/svm.model', 'wb'))
#���һ�䱣��ģ�ͣ��Ժ����ͨ������������¼���ģ�ͣ�
#model = pickle.load(open('../tmp/svm.model', 'rb'))

#���������صĿ⣬���ɻ�������
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train)) #ѵ�������Ļ�������
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test)) #���������Ļ�������

#������
pd.DataFrame(cm_train, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile2)
