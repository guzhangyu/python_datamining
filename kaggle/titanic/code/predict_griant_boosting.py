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

def split_constant_cols(data,col_name):
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer(sparse=False, dtype=int)
    di = []
    # for i in data['Embarked']:
    #     if i and type(i) == str:
    #         di.append({'Embarked': i})
    #         # pass
    #     else:
    #         # di.append({'Embarked': None})
    #         pass
    # data1=vec.fit_transform(di)
    # print(len(data1))

    for i in data[col_name].index:
        if data[col_name][i] and type(data[col_name][i]) == str:
            di.append({col_name: data[col_name][i], 'index': i})

    data1 = vec.fit_transform(di)
    data1 = pd.DataFrame(data1[:, :-1], index=data1[:, -1])
    for i in data1:
        data[col_name + str(i)] = data1[i]
        data[col_name + str(i)] = data[col_name + str(i)].fillna(data[col_name + str(i)].mean())

def read_data(filename,columns):
    data = pd.read_csv(filename, index_col='PassengerId',usecols=columns)
    data['Sex'] = data['Sex'].apply(lambda a: 1 if a == 'male' else 0)
    data=split_constant_cols(data,'Embarked')
    # data['Embarked'] = data['Embarked'].apply(lambda a: 1 if a == 'S' else 2 if a == 'C' else 3)
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

all_data=pd.concat([data,data2],sort=False)
data=fill_d(data,all_data)

data2=fill_d(data2,all_data)
# data['Age'] = data['Age'].fillna(data['Age'].mean())
all_data=pd.concat([data,data2])
data2['Age'] = data2['Age'].fillna(all_data['Age'].mean())

# data=data.dropna()
data['Age'] = data['Age'].fillna(all_data['Age'].mean())
data=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived'])
x=data.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
y=data['Survived']

data2=data2.reindex(columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

from sklearn.ensemble import GradientBoostingClassifier

# rf=GradientBoostingClassifier(max_features=5,n_estimators=40,max_depth=6,min_samples_leaf=13,min_samples_split=38,subsample=0.75)#n_estimators=140,max_depth=4,min_samples_split=50
# param_test1 = {}# 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9] 'min_samples_leaf':range(1,20,3), 'min_samples_split':range(2,100,6) 'max_depth':range(3,14,1), 'min_samples_split':range(2,100,6) 'n_estimators':range(10,300,10)
# gsearch1=GridSearchCV(estimator=rf,param_grid=param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(x,y)
# #gsearch1.cv_results_,
# print(gsearch1.best_estimator_,gsearch1.best_params_, gsearch1.best_score_)
#
# exit()

def model_param_optimize(model,params,x,y):
    from sklearn.model_selection import GridSearchCV
    param_test1 = {}
    for param in params:  # max_features=5,n_estimators=40,max_depth=6,min_samples_leaf=13,min_samples_split=38,subsample=0.75
        # n_estimators=140,max_depth=4,min_samples_split=50
        gsearch1 = GridSearchCV(estimator=model, param_grid=param, scoring='roc_auc', iid=False, cv=5)
        gsearch1.fit(x, y)
        # gsearch1.cv_results_,
        print(gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_)
        param_test1.update(gsearch1.best_params_)

        for p in param:
            if p == 'min_samples_split' and 'max_depth' in param:
                continue
            setattr(model, p, gsearch1.best_params_[p])

    print(param_test1)
    return model

rf=GradientBoostingClassifier()
params=[{'n_estimators':range(10,300,10)},
        { 'min_samples_split':range(2,100,6), 'max_depth':range(3,14,1)},
        {'min_samples_leaf':range(1,20,3), 'min_samples_split':range(2,100,6)},
        {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]},
        {'max_features':range(1,8)}]
rf=model_param_optimize(rf,params,x,y)
rf=model_param_optimize(rf,[{'learning_rate':np.arange(0.1,0,-0.01),'n_estimators':range(rf.n_estimators,rf.n_estimators*10+1,rf.n_estimators)}],x,y)
# exit()


# rf=GradientBoostingClassifier(max_features=5,n_estimators=40,max_depth=6,min_samples_leaf=13,min_samples_split=38,subsample=0.75)
rf.fit(x,y)
# yp=rf.predict(x)
# # print(rf.estimators_)
# # print(len(rf.estimators_))
#
# from sklearn import metrics
#
# print("Accuracy : %.4g" % metrics.accuracy_score(y,yp))
# yp_prob=rf.predict_proba(x)[:,1]
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y,yp_prob))
# exit()
# r=r.astype(int).reshape((1,-1))[0]
# print(np.unique(r))

yp=rf.predict(data2)
pd.Series(yp,index=data2.index).to_csv('../data/result3.csv')
