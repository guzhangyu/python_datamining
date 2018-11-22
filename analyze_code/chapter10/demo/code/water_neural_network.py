#-*- coding: utf-8 -*-
#建立、训练多层神经网络，并完成模型的检验
from __future__ import print_function
import pandas as pd

inputfile1='../data/train_neural_network_data.xls' #训练数据
inputfile2='../data/test_neural_network_data.xls' #测试数据
testoutputfile = '../tmp/test_output_data.xls' #测试数据模型输出文件
data_train = pd.read_excel(inputfile1) #读入训练数据(由日志标记事件是否为洗浴)
data_test = pd.read_excel(inputfile2) #读入测试数据(由日志标记事件是否为洗浴)

use_water=data_train.iloc[:,-4][data_train.iloc[:,4]==1]
print(use_water)
describe=use_water.describe(percentiles=[0.1,0.9])

del describe['count']
del describe['std']

print(describe)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
use_water_df=pd.DataFrame(use_water)
# use_water_df.boxplot(sym='+',showmeans=True,meanline=True)
# use_water_df.plot()

fig=plt.figure(3)
ax1=fig.add_subplot(311)
# use_water_df.boxplot(ax=ax1,sym='+',showmeans=True,meanline=True,vert=False)
ax1.boxplot(use_water,vert=False)
plt.xticks(describe.values.tolist())
ax1.grid(True)

# ax1.plot(use_water)

ax1=fig.add_subplot(312)
use_water_df.hist(ax=ax1,grid=True,density=True,stacked=True)

# ax1.hist(use_water,density=True,stacked=True)

ax1=fig.add_subplot(313)
ax1.scatter(use_water,[1]*len(use_water))

# use_water_df.hist()
# ax1.hist(use_water)
# plt.xticks()
plt.show()
exit()

y_train = data_train.iloc[:,4].as_matrix() #训练样本标签列
x_train = data_train.iloc[:,5:17].as_matrix() #训练样本特征
y_test = data_test.iloc[:,4].as_matrix() #测试样本标签列
x_test = data_test.iloc[:,5:17].as_matrix() #测试样本特征

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(11, 17)) #添加输入层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(17, 10)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(10, 1)) #添加隐藏层、输出层的连接
model.add(Activation('sigmoid')) #以sigmoid函数为激活函数
#编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) #class_mode = 'binary'


model.fit(x_train, y_train, nb_epoch = 100, batch_size = 1) #训练模型
model.save_weights('../tmp/net.model') #保存模型参数

r = pd.DataFrame(model.predict_classes(x_test), columns = [u'预测结果'])
pd.concat([data_test.iloc[:,:5], r], axis = 1).to_excel(testoutputfile)
model.predict(x_test)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果
