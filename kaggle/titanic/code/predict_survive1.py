import pandas as pd

def read_data(filename,columns):
    data = pd.read_csv(filename, index_col='PassengerId',usecols=columns)
    data['Sex'] = data['Sex'].apply(lambda a: 1 if a == 'male' else 0)
    data['Embarked'] = data['Embarked'].apply(lambda a: 1 if a == 'S' else 2 if a == 'C' else 3)
    return data

data=read_data('../data/train.csv',['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])
data=data.dropna()

x=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
# for a in x.columns:
#     x[a]=(x[a]-x[a].mean())/x[a].std()
y=data['Survived']

# import numpy as np
# pd.Series(np.zeros(len(data)),index=data.index).to_csv('../data/result.csv')
# exit()

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 7, output_dim = 14))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim = 14, output_dim = 1))
model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) #class_mode = 'binary'
#编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
#另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
#求解方法我们指定用adam，还有sgd、rmsprop等可选
model.fit(x, y, nb_epoch = 1000, batch_size = 16) #训练模型，学习一千次

data2=read_data('../data/test.csv',['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])
yp = model.predict_classes(data2).reshape(len(data2)) #分类预测


# print(yp)
pd.Series(yp,index=data2.index).to_csv('../data/result.csv')
