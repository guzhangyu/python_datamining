import numpy as np
import pandas as pd

def find_rule(data_frame, support, confidence, min_size=2, ms =u'--'):
    max_filled_len=np.max(np.sum(data_frame, axis=1)).astype(int)
    value_len = len(data_frame.values)
    print(value_len)
    columns = data_frame.columns
    success_data=[]
    data_frame_cur=data_frame
    for i in range(0,max_filled_len):
        ds=data_frame_cur.sum(axis=1)>=i+1
        if not np.any(ds) or ds.sum()<value_len*support:
            break
        data_frame_cur = data_frame_cur.reindex(data_frame_cur.index[ds.values.tolist()])
        # csr=np.unique(data_frame_cur, axis=0).astype(int)
        # csr1=[csri for csri in csr
        #                 if np.all(data_frame[
        #                               [data_frame.columns[i] for i in range(0,len(csri)) if csri[i]==1]
        #                           ]==1,axis=1).sum()/value_len>=support]
        # cs=[data_frame.columns[c==1].values.tolist() for c in csr1]
        # print(cs)
        # for c in cs:
        combines = get_combines(i,columns,data_frame_cur)
        columns_pair = gen_columns_pair(combines, success_data, data_frame, support, confidence, value_len)

        if not columns_pair:
            break

        columns_new=set()
        for a in columns_pair:
            columns_new=columns_new.union(set(a))
        columns=list(columns_new)
        data_frame_cur=data_frame_cur[columns]

    for data in success_data:
        print(ms.join(list(data[0])+[data[1]]),data[2],data[3])
    print("end")

    return success_data


def gen_columns_pair(combines, success_data, data_frame, support, confidence, value_len):
    columns_pair = []
    print(combines)
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


def get_combines(i, y,dataframe):
    yy = set([tuple(y[:i + 1])])
    # print("yy:")
    # print(yy)
    for j in range(i + 1, len(y)):
        new_yy = set()
        for yyy in yy:
            for k in range(0, len(yyy)):
                new_yy.add(tuple(yyy[0:k] + yyy[k + 1: len(yyy)+1]))
        for yyy in new_yy:
            yy.add(yyy+ tuple([y[j]]))
    return yy

# x=get_combines(2,['a','b','e','c'])
# print(x)

#r=find_rule(pd.DataFrame([[0,1,1],[1,0,1]],columns=['xa','y','z']) ,0.06,0.7)
# print(r)