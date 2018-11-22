import pandas as pd

data=pd.read_excel('model.xls')
columns=data.columns
data=data.values
x=data[:,:3]
y=data[:,3]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim = 10, output_dim = 1))
model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) #class_mode = 'binary'
#编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
#另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
#求解方法我们指定用adam，还有sgd、rmsprop等可选

model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 16) #训练模型，学习一千次
yp = model.predict_classes(x_test).reshape(len(y_test)) #分类预测

score=model.evaluate(x_test,y_test,batch_size=16)
print(score)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果