import pandas as pd

data=pd.read_csv('../data/train.csv',index_col='PassengerId',usecols=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])#
data['Sex']=data['Sex'].apply(lambda a:1 if a=='male' else 0)
data['Embarked']=data['Embarked'].apply(lambda a:1 if a=='S' else 2 if a=='C' else 3)
# data['Age']=data['Age'].fillna(data['Age'].mean())
data=data.dropna()
data=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])
data['Survived']=data['Survived'].astype(float)

from sklearn.tree import DecisionTreeClassifier #导入决策树模型
from sklearn.model_selection import train_test_split

x=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
# for a in x.columns:
#     x[a]=(x[a]-x[a].mean())/x[a].std()
y=data['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)

from LassoModel import LassoModel
from random import shuffle

idx=data.index.values
shuffle(idx) #随机打乱数据
data.reindex(idx)
p = 0.8 #设置训练数据比例
train = data.iloc[:int(len(data)*p),:] #前80%为训练集
test = data.iloc[int(len(data)*p):,:] #后20%为测试集

model = LassoModel(train)
model.glmnet(train)
# model.fit(train)
# model.gray(train)
model.deep_learn_train1(train)
r=model.predict_data_binary(test)
r=r.astype(int).reshape((1,-1))[0]
print(r)

def cm_plot(true_data,predicted):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    cm = confusion_matrix(true_data,predicted)  # 混淆矩阵

    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()  # 显示作图结果

# cm_plot(test.iloc[:,-1],r)
# exit()
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.linear_model import RandomizedLogisticRegression as RLR
# rlr = RLR() #建立随机逻辑回归模型，筛选变量
# rlr.fit(x_train, y_train) #训练模型
# rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
# print(u'通过随机逻辑回归模型筛选特征结束。')
# print(u'有效特征为：%s' % ','.join(data.columns[:7][rlr.get_support()]))
# print(rlr.get_support())
# # x = data[data.columns[:-1][rlr.get_support()]].values #筛选好特征
# print('rlr : %s' % rlr.all_scores_)
#
# lr = LR() #建立逻辑回归模型
# lr.fit(x_train, y_train) #用筛选后的特征数据来训练模型
# print(u'逻辑回归模型训练结束。')
# print(u'模型的平均正确率为：%s' % lr.score(x_train, y_train)) #给出模型的平均正确率，本例为81.4%
# y_test1=lr.predict(x_test)
# cm_plot(y_test,y_test1)
# exit()

tree = DecisionTreeClassifier() #建立决策树模型
tree.fit(x_train,y_train) #训练

# cm_plot(y_train, tree.predict(x_train))
# cm_plot(y_test, tree.predict(x_test))
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

model.fit(x_train, y_train, nb_epoch = 1500, batch_size = 16) #训练模型，学习一千次
yp = model.predict_classes(x_test).reshape(len(y_test)) #分类预测

score=model.evaluate(x_test,y_test,batch_size=16)
print(score)
cm_plot(y_test,yp)

#保存模型
# from sklearn.externals import joblib
# joblib.dump(tree, treefile)


