import pandas as pd
import numpy as np
import re

re_str=', (?:Dr|M(?:aster|rs|r|iss|s))\. \(?'
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
data['Survived']=data['Survived'].astype(float)

from LassoModel import LassoModel
train=data
model = LassoModel(train)
model.glmnet(train)
# model.fit(train)
# model.gray(train)
model.deep_learn_train1(train)

data2=read_data('../data/test.csv',['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                'Embarked'])
r=model.predict_data(data2)
r=r.astype(int).reshape((1,-1))[0]
import numpy as np
print(np.unique(r))
pd.Series(r,index=data2.index).to_csv('../data/result2.csv')
