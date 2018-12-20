import pandas as pd
import numpy as np
import re

re_str=', (?:Dr|M(?:aster|rs|r|iss))\. \(?'
family_name=re.compile(re_str+'(\w+)')

def read_data(filename,columns):
    data = pd.read_csv(filename, index_col='PassengerId',usecols=columns)
    data['Sex'] = data['Sex'].apply(lambda a: 1 if a == 'male' else 0)
    data['Embarked'] = data['Embarked'].apply(lambda a: 1 if a == 'S' else 2 if a == 'C' else 3)

    b=data[data['Age'].isnull()]
    names=b #[b['SibSp']>0]
    ls=pd.DataFrame()
    for i in names.index:
        fn=re.search(family_name,names['Name'][i])
        if not fn:
            print(names['Name'][i])
            continue
        fn=fn.group(1)
        others=data[data['Name'].str.contains(re_str+fn)]
        # print(others['Age'].dropna().index)

        others=others.ix[others['Age'].dropna().index.values,:]
        pre_others = others.copy()
        if len(others)>1:
            others1=others[others['SibSp']==names['SibSp'][i]]
            others2 = others[others['Parch'] == names['Parch'][i]]

            others=pd.DataFrame(columns=others.columns)
            if names['SibSp'][i]>0:
                others=others.append(others1)
            if names['Parch'][i]>0:
                others=others.append(others2)
            # len(others) == 0 说明前两者中>0的都=0，即如果n-SibSp>0，只能取SibSp!=n-SibSp的数据
            # 此时，如果两者中有=0的，还可以取到某一项相等的项
            if len(others) == 0:
                if names['SibSp'][i]==0 and names['Parch'][i]==0:
                    others = others1[others1['Parch']==0]

                if len(others) == 0:
                    if names['SibSp'][i]==0:
                        others=others1
                    elif names['Parch'][i]==0:
                        others=others2

            #如果仍然len(others) == 0,说明names两者都不为0，此时取不到一致的数据了
            # if len(others)==0:
            #     others=others1.append(others2)

            if len(others) == 0:
                others = pre_others

            if len(others)!=1:
                others1=others[others['Embarked']==names['Embarked'][i]]
                if len(others1)>0:
                    others=others1

                if len(others) > 1:
                    others1 = others[others['Pclass'] == names['Pclass'][i]]
                    if len(others1) > 0:
                        others = others1
            others = others.drop_duplicates()

        if len(pre_others)>0:
            # print(others)
            if len(others) == 0:
                # print(i, fn, "====1")
                # print(names.ix[i, :].copy())
                # print("====2")
                # print("before:")
                # print(pre_others)
                pass
            else:
                data.ix[i,'Age'] =others['Age'].mean()
        else:
            ls=ls.append(data.ix[i,:])
    data['Age']=data['Age'].fillna(data['Age'].mean())

    # print(ls)
    d=data.isnull().sum()
    # print("','".join(d[d>0].index))
    print(d[d>0])
    # for i in d[d>0].index:
    #     print("%s:" % i)
    #     print(np.unique(data[i]))
    return data

data=read_data('../data/train.csv',['Name','PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])

data=data.dropna()
data=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])
from sklearn.model_selection import train_test_split

x=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
y=data['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=4,n_estimators=100,random_state=1,oob_score=True)
rf.fit(x_train,y_train)
yp=rf.predict(x_test)

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


# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
#
# model = Sequential() #建立模型
# model.add(Dense(input_dim = 7, output_dim = 14))
# model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
# model.add(Dense(input_dim = 14, output_dim = 1))
# model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数
#
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) #class_mode = 'binary'
# #编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
# #另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
# #求解方法我们指定用adam，还有sgd、rmsprop等可选
# model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 16) #训练模型，学习一千次
#
# # data2=read_data('../data/test.csv',['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
# #                                 'Embarked'])
# yp = model.predict_classes(x_test).reshape(len(y_test)) #分类预测

cm_plot(y_test,yp)