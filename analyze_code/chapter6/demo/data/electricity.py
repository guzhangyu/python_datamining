import pandas as pd

data=pd.read_excel('model.xls')
columns=data.columns
data=data.values
x=data[:,:3]
y=data[:,3]

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)

dtc.fit(x_train, y_train) #训练模型
yp=dtc.predict(x_test)
print(yp)
score=dtc.score(x_test,y_test)
print(score)

from cm_plot1 import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果

#导入相关函数，可视化决策树。
#导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# with open("tree.dot", 'w') as f:
#   f = export_graphviz(dtc, feature_names = columns[:-1], out_file = f) #dot -Tpdf filename.dot -o outfile.pdf
