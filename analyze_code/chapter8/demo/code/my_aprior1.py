import numpy as np
import pandas as pd

def find_rule(data_frame, support, confidence, min_size=2, ms =u'--'):
    columns = data_frame.columns.values
    columns=columns.reshape((len(columns), 1)).tolist()
    success_data=[]
    while len(columns)>0:
        columns = get_combines(columns)
        columns = gen_columns_pair(columns, success_data, data_frame, support, confidence)

    for data in success_data:
        print(ms.join(list(data[0])+[data[1]]),data[2],data[3])
    print("end")

    return success_data

#这一步已经保证了导出结果列的遍历
def gen_columns_pair(combines, success_data, data_frame, support, confidence):
    columns_pair = []
    # print(combines)
    value_len=len(data_frame)
    for y1 in combines:
        # print("y1:")
        # print(y1)
        s = np.all(data_frame[list(y1)] == 1, axis=1).sum()
        if s / value_len >= support:
            columns_pair.append(y1)
        else:
            continue

        if len(y1) <= 1:
            continue

        for a in y1:
            left = set(y1) - set([a])
            confidence_1 = s / np.all(data_frame[list(left)] == 1, axis=1).sum()
            if confidence_1 >= confidence:
                success_data.append([left, a, s / value_len, confidence_1])
    return columns_pair


def get_combines(x):
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]])) #sorted 保证了序列的有序
    return r


# x=get_combines(2,['a','b','e','c'])
# print(x)

r=find_rule(pd.DataFrame([[0,1,1],[1,0,1]],columns=['xa','y','z']) ,0.06,0.7)
# print(r)