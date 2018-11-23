import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql://root:root@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('select * from a', engine, chunksize = 10000)

for i in sql:
    print(i)
    i.to_sql('b',engine,if_exists='append')

