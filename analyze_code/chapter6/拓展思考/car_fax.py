import pandas as pd

data=pd.read_excel('拓展思考样本数据.xls',index_col=0)

# print(data['输出'].values=='正常')
columns=list(set(data.columns.values)-{'销售类型','销售模式','输出'})
x=data[columns]
print("-----------")
print(x.mean())
print(x.std())
print(x)
x=(x-x.mean())/x.std()
print(x)
y=data['输出']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)

# print(x_train,x_test,y_train,y_test)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x_train, y_train) #训练模型
yp=dtc.predict(x_test)
score=dtc.score(x_test,y_test)
print(score)

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果

#导入相关函数，可视化决策树。
#导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open("tree.dot", 'w') as f:
  f = export_graphviz(dtc, feature_names = columns, out_file = f) #dot -Tpdf filename.dot -o outfile.pdf

#dot -Tpng tree.dot>tree.png