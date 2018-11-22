from math import nan

import pandas as pd
import numpy as np


def count_by_element(element, arr):
    element=np.array(element)
    print(element)
    for i in element:
        index1=''.join(element[element != i])

        d = arr.get(index1)
        if not d:
            d = {}
            arr.update({index1: d})

        d.update({i: d.get(i, 0) + 1})
    print(arr)

def count_contains(l,s):
    ct=0
    for el in l:
        all_in=True
        for es in s:
            if es not in el:
                all_in=False
                break
        if all_in:
            ct=ct+1
    return ct

print("--------------------------------------------------")

data=pd.read_excel('menu_orders.xls', header = None)
data_v=data.values
# a=set(v1.flatten())-{np.NAN}
# a=sorted(list(a))
# print("all elements: "+a)
cts=np.count_nonzero(data.notnull().values,1)

for k in range(1,max(cts)):
    cur_data_v=data_v[cts >= k + 1]
    arr = {}
    print("cal for ==========================="+str(k))
    for data_v_line in cur_data_v:
        data_v_line = [a for a in data_v_line if type(a) == str]

        elements = set([tuple(data_v_line[0:k + 1])])
        for i in range(k+1, len(data_v_line)):
            cur_elements = set()
            #print(elements)
            for ce in elements: #将原先的元素依次减掉1个元素的方法，来得到所有k-1个元素的组合
                #print([ce[0:i] + ce[i + 1: len(ce)] for i in range(1, len(ce) - 1)])
                cur_elements=cur_elements.union(set([ce[0:i] + ce[i + 1:len(ce)+1] for i in range(0, len(ce))]))
            #print(cur_elements)

            for ce in cur_elements:
                #print(elements, ce, data_v_line)
                elements.add(ce + tuple(data_v_line[i:i + 1])) #追加末尾元素

        for ce in elements: # 在这个层级，因为每一行对elements是会重复的，但是要反复加到arr中
            count_by_element(ce, arr)
    for arr_k in arr:
        arr_v=arr[arr_k]
        for cur_k in arr_v:
            cur_v=arr_v[cur_k]
            arr_v.update({cur_k:[round(cur_v/len(data_v),3),round(cur_v/count_contains(data_v,arr_k),3)]})
    print(arr)
            # ct=i-k
            # for kkkk in range(1,ct+1):
                # for kkkkk in range(k+kkkk,0,-1):
                #     a1=[]
                #     for ii in range(0,kkkk):
                #         if ii!=kkkk:
                #             a1.append(l[ii])
                #     for ii in range(k+kkkk+1,i+1):
                #
                #     ar.add(a1)
                #
                # for kkk in range(i,k+1,-1):


    #         index1 = ''.join(l[i-k:i])
    #         d = arr.get(index1, {})
    #         if not d:
    #             arr.update({index1: d})
    #         d = arr.get(index1, {})
    #         if not d:
    #             arr.update({index1: d})
    #
    #         for j in range(i + 1, len(l)):
    #             index2 = l[j]
    #             d.update({index2: d.get(index2, 0) + 1})
    #
    # for i in arr:
    #     for j in arr[i]:
    #         arr[i][j] = [round(arr[i][j] / len(v1), 3), round(arr[i][j] / np.count_nonzero(v == j), 3)]
    # print(arr)




# count1(np.array(['1','2','3']))
# v=v1[cts>=1]
# arr={}
# for l in v:
#     l=[a for a in l if type(a)==str]
#     for i in range(0,len(l)-1):
#         index1=l[i]
#         d = arr.get(index1, {})
#         if not d:
#             arr.update({index1:d})
#
#         for j in range(i+1,len(l)):
#             index2=l[j]
#             d.update({index2:d.get(index2,0)+1})
# for i in arr:
#     for j in arr[i]:
#         arr[i][j]=[round(arr[i][j]/len(v1),3),round(arr[i][j]/np.count_nonzero(v==index1),3)]
# print(arr)
#
# arr={}
# for index1 in a:
#     for index2 in a:
#         s=0
#         for k in v:
#             kk=k.tolist()
#             vvv=1 if(kk.count(index1)>0 and kk.count(index2)>0 and kk.index(index2)>kk.index(index1))else 0
#             s=s+vvv
#         if s>0:
#             # print(s/len(v),s/np.count_nonzero(v==index1))
#             d=arr.get(index1,{})
#             if not d:
#                 arr.update({index1:d})
#             d.update({index2:[round(s/len(v),3),round(s/np.count_nonzero(v==index1),3)]})
# print(arr)