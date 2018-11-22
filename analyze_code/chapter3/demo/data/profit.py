import pandas as pd

data=pd.read_excel('catering_dish_profit.xls',index_col=0)['盈利']

# print(data.keys())
# print(data.values)

# f=data.values
# t=f.sum()
# f=f/t
# print(f)
# f=f.cumsum()
# print(f)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
# ax.bar(data.keys(),data.values)
# ax.plot(data.keys(),f)
plt.figure()
data.plot(kind='bar')
(data.cumsum()/data.sum()).plot(secondary_y = True)

plt.show()
#print(data)