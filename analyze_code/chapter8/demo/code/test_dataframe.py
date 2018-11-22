import pandas as pd
import numpy as np
df=pd.DataFrame([[1,2],[2,3],[1,2]],columns=['a','b'])
# print(df.reindex([1]))
# print(df.T)
# print(df)
# print(np.unique(df,axis=0))
# print(df.nunique(axis=0))


# print(df.reindex(df.index[[True,True,True]]))
# print(np.all(df>=2,axis=1))
# a=[tuple([5])]
# print(set(set(e) for e in a))

# print(df.cumsum())
# print(pd.concat([df,df.cumsum(),df-df.cumsum()-1]))
# print(df.diff())
# df['a']=df['a'].apply(lambda a:1)
# print(df)

df.index.reindex()