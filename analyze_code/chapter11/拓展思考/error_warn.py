import pandas as pd
import numpy as np
data=pd.read_excel('拓展思考样本数据.xls',index_col='日期')
# print([1 if a>0 else 0 for a in data.iloc[:,-1].values])
# print([[1 if a>0 else 0 for a in b]for b in data.iloc[:,:-1].values])
# x=[[1 if a>0 else 0 for a in b]for b in data.iloc[:,:-1].values]
# y=np.array([1 if a>0 else 0 for a in data.iloc[:,-1].values]).reshape((-1,1))
#   #data.iloc[:,-1].values.reshape((-1,1))
# print(y)
data=data.apply(lambda a: [1 if i>0 else 0 for i in a],axis=0,broadcast=True)


# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)
#
# from sklearn.tree import DecisionTreeClassifier as DTC
# dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
# dtc.fit(x_train,y_train)
# y_predicted=dtc.predict(x_test)
# # print(y_test,y_predicted.reshape((-1,1)))
# print("score:%f" % dtc.score(x_test,y_test))

#导入相关函数，可视化决策树。
#导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# with open("tree.dot", 'w') as f:
#   f = export_graphviz(dtc, feature_names = data.columns[:-1], out_file = f) #dot -Tpdf filename.dot -o outfile.pdf

#dot -Tpng tree.dot>tree.png


from apriori import Apriori
# data=np.append(x,y,axis=1)
data=data.apply(lambda a: [1 if i>0 else 0 for i in a],axis=0,broadcast=True)
print(data)
apriori=Apriori(data,0.06)
result=apriori.find_rule(data,0.06,0.75)
result=result.reindex(result.index[[ a.endswith('故障类别') for a in result.index]])
print('after:')
print(result)
a=result.index.values
b=set()
for i in a:
  b=b.union(set(i.split('--')))
print(b)

from arima_model1 import ArimaModel
print(data.iloc[:,0])
am=ArimaModel(data['日志类告警'],0.3)
am.fit()