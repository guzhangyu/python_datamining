import pandas as pd

data=pd.read_excel('data.xls')

# print(data['输出'].values=='正常')
# print(data.columns.values)
columns=list(set(data.columns.values)-{'TNM分期','转移部位','病程阶段','确诊后几年发现转移'})
# x=data[columns]

# for column in ['病程阶段','确诊后几年发现转移']:
#     # print("-=-=-=-=-=")
#     # print(x[column])
#     x[column]=[float(a[1:]) for a in x[column]]
#         #map(lambda a:int(a[1:]),x[column])
#     # print("xxxxxxxxxxxxx-=-=-=-=-=")
#     # print(x[column])
# # columns1=list(set(data.columns.values)-{})
#
# x=(x-x.mean())/x.std()
# # print(x)
column_arr={
'肝气郁结证型系数':'A',
'热毒蕴结证型系数':'B',
'冲任失调证型系数':'C',
'气血两虚证型系数':'D',
'脾胃虚弱证型系数':'E',
'肝肾阴虚证型系数':'F'
}

from sklearn.cluster import KMeans
import numpy as np

# km = KMeans(n_clusters=4).fit(data[columns])
# print(km.cluster_centers_)

rr=[]
for a in columns:
    xx=data[a].values
    km=KMeans(n_clusters=4).fit(np.reshape(xx,(len(xx),1)))
    data[a]=km.labels_+1


    means=pd.DataFrame(km.cluster_centers_).sort_values(0).rolling(2).mean().values
    # print(means)
    means=means.reshape((1,4)).tolist()[0]
    means=sorted(means)
    # print(means)
    means[0]=0
    rr.append(means)
    b = [0, 0, 0, 0]
    pre=0
    for i in range(3,-1,-1):
        b[i]=np.count_nonzero(xx>=means[i])-sum(b[i:])
        now_pre=np.count_nonzero(xx>=km.cluster_centers_[i])
        # print(now_pre-pre)
        pre=now_pre

        x1=set([k for k in range(0,len(km.labels_)) if km.labels_[k]==i])
        x2=set([k for k in range(0,len(xx)) if xx[k]>=means[i] and (i==3 or xx[k]<means[i+1])])
        print(i)
        print(means)
        print(min(xx[list(x2)]),max(xx[list(x2)]))
        print(x1-x2)
        print(x2 - x1)


    print("b:")
    print(b)
    # rr.append(b)
    # rr.append(km.cluster_centers_.reshape((1, 4)).tolist()[0])
    # rr.append(b)

    h=pd.Series(km.labels_).value_counts()
    h=pd.concat([pd.DataFrame(km.cluster_centers_,columns=['0']),pd.DataFrame(h,columns=['1'])],axis=1).sort_values('0')
    print(h)
    # h=h.astype(int).tolist()
    # print(h)
    rr.append(h['1'].values.tolist())

print(rr)
result = pd.DataFrame(data=rr,index=['A','An','B','Bn','C','Cn','D','Dn','E','En','F','Fn'],columns=range(0,4))
print(result)
result.to_excel('disease.xls')




y=data['TNM分期']
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)
#
# # print(x_train,x_test,y_train,y_test)
#
# from sklearn.tree import DecisionTreeClassifier as DTC
# dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
# dtc.fit(x_train, y_train) #训练模型
# yp=dtc.predict(x_test)
# score=dtc.score(x_test,y_test)
# print(score)
#
# from cm_plot import * #导入自行编写的混淆矩阵可视化函数
# cm_plot(y_test,yp).show() #显示混淆矩阵可视化结果
#
# #导入相关函数，可视化决策树。
# #导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# with open("tree.dot", 'w') as f:
#   f = export_graphviz(dtc, feature_names = columns, out_file = f) #dot -Tpdf filename.dot -o outfile.pdf
#
# #dot -Tpng tree.dot>tree.png