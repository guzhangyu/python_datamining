#-*- coding: utf-8 -*-
#建立、训练多层神经网络，并完成模型的检验
from __future__ import print_function
import pandas as pd

inputfile1='../data/train_neural_network_data.xls' #训练数据
inputfile2='../data/test_neural_network_data.xls' #测试数据
testoutputfile = '../tmp/test_output_data.xls' #测试数据模型输出文件
data_train = pd.read_excel(inputfile1) #读入训练数据(由日志标记事件是否为洗浴)
data_test = pd.read_excel(inputfile2) #读入测试数据(由日志标记事件是否为洗浴)
y_train = data_train.iloc[:,4].as_matrix() #训练样本标签列
x_train = data_train.iloc[:,5:17].as_matrix() #训练样本特征


def split_washes(data):
  result = pd.DataFrame(data)
  result = result.apply(lambda a: (0 if a[1] <= 900 else (a[1] - 900) * 0.005) +
                                  (0 if a[4] <= 10 else (a[4] - 10) * 0.5) +
                                  (0 if a[6] >= 0.5 else (0.5 - a[6]) * 0.2) +
                                  (0 if a[7] <= 30 else (a[7] - 30) * 0.2) +
                                  (0 if a[10] <= 1000 else (a[10] - 1000) * 0.002)
                        , axis=1)
  result = result > 5
  # [ for i in result]
  index=[a for a in range(0,len(result)) if result[a]]
  print(index)
  for a in data[result]:
    for d in a:
      print("%.2f" %d)
  exit(0)




y_test = data_test.iloc[:,4].as_matrix() #测试样本标签列
x_test = data_test.iloc[:,5:17].as_matrix() #测试样本特征
print(data_test)
split_washes(x_test)
from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x_train, y_train) #训练模型
yp=dtc.predict(x_test)
score=dtc.score(x_test,y_test)
print(score)

# #导入相关函数，可视化决策树。
# #导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# with open("tree.dot", 'w') as f:
#   f = export_graphviz(dtc, feature_names = data_train.columns[5:17], out_file = f) #dot -Tpdf filename.dot -o outfile.pdf
#
# #dot -Tpng tree.dot>tree.png

# exit()

from cm_plot import * #导入自行编写的混淆矩阵可视化函数
cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果

