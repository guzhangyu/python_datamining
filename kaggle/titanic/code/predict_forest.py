import pandas as pd
import numpy as np
import re

re_str=', (?:Dr|M(?:aster|rs|r|iss|s))\. \(?'
family_name=re.compile(re_str+'(\w+)')

def minus_others_by_relative(others,names,i):
    others1 = others[others['SibSp'] == names['SibSp'][i]]
    #认为>0的前提下相同更加精准
    if names['SibSp'][i] > 0: #说明是同辈
        if len(others1)>0:
            jo = others1[others1['Parch'] == names['Parch'][i]] #说明是夫妻
            if len(jo)>0:
                return jo,1
            return others1,1

    others2 = others[others['Parch'] == names['Parch'][i]]
    #正常情况下，下面都是长辈或者子辈
    if names['Parch'][i]>0:
        if names['Parch'][i] == 1:  # 如果对方也=1，说明是我的子辈或父辈，对方的Parch>1的情况，在上面就应该匹配到了
            if len(others2) > 0:
                return others2, 2

    # 前两者中>0的都=0，即如果n-SibSp>0，只能取SibSp!=n-SibSp的数据
    # 此时，如果两者中有=0的，还可以取到某一项相等的项
    #此时 others1 或者 others2 有数据，说明n-SibSp=0

    # 如果仍然len(others) == 0,说明names两者都不为0，此时取不到一致的数据了
    # if len(others)==0:
    #     others=others1.append(others2)

    return others,1

def minus_others(others,names,i):
    others,level=minus_others_by_relative(others,names,i)

    if len(others) > 1:
        others1 = others[others['Embarked'] == names['Embarked'][i]]
        if len(others1) > 0:
            others = others1

        if len(others) > 1:
            others1 = others[others['Pclass'] == names['Pclass'][i]]
            if len(others1) > 0:
                others = others1
    # others = others.drop_duplicates()
    return others,level

def read_data(filename,columns):
    data = pd.read_csv(filename, index_col='PassengerId',usecols=columns)
    data['Sex'] = data['Sex'].apply(lambda a: 1 if a == 'male' else 0)
    data['Embarked'] = data['Embarked'].apply(lambda a: 1 if a == 'S' else 2 if a == 'C' else 3)
    return data

def fill_d(data,all_data):
    b = data[data['Age'].isnull()]
    names = b  # [b['SibSp']>0]
    ls = pd.DataFrame()
    for i in names.index:
        fn = re.search(family_name, names['Name'][i])
        if not fn:
            print(names['Name'][i])
            continue
        fn = fn.group(1)
        others = all_data[all_data['Name'].str.contains(re_str + fn)]

        # Age不为null的数据
        others = others.ix[others['Age'].dropna().index.values, :]
        level=1
        pre_others = others.copy()
        if len(others) > 1:
            others,level = minus_others(others, names, i)

        if len(pre_others) > 0:
            # print(others)
            if len(others) == 0:
                # print(i, fn, "====1")
                # print(names.ix[i, :].copy())
                # print("====2")
                # print("before:")
                # print(pre_others)
                pass
            else:
                am = others['Age'].mean()
                if level == 2 and am < 18:
                    am = am * 2  # 此时任务当前为这些人的父母
                data.ix[i, 'Age'] = am
        else:
            ls = ls.append(data.ix[i, :])

    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    # # print(ls)
    # d = data.isnull().sum()
    # # print("','".join(d[d>0].index))
    # print(d[d > 0])
    # # for i in d[d>0].index:
    # #     print("%s:" % i)
    # #     print(np.unique(data[i]))
    return data

data=read_data('../data/train.csv',['Name','PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])
data2=read_data('../data/test.csv',['Name','PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])

all_data=pd.concat([data,data2])
data=fill_d(data,all_data)
data2=fill_d(data2,all_data)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data2['Age'] = data2['Age'].fillna(data2['Age'].mean())


data=data.dropna()
data=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])
x=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
y=data['Survived']

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=4,n_estimators=100,random_state=1,oob_score=True)
rf.fit(x,y)

data2=data2.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
r=rf.predict(data2)
# r=r.astype(int).reshape((1,-1))[0]
# print(np.unique(r))
pd.Series(r,index=data2.index).to_csv('../data/result2.csv')
