from pyspark import SparkContext
from pyspark.sql.context import SQLContext

sc=SparkContext()
sqlContext=SQLContext(sc)

rdd=sc.textFile('hdfs://master:8020/secondary_sort/input/sample_input.txt')
def date_temp_f(s):
    s=s.split(',')
    return s[0]+'-'+s[1],int(s[3])

rdd=rdd.map(date_temp_f)

rdd=rdd.groupByKey()
rdd.map(lambda x:(x[0],','.join([str(i) for i in sorted(x[1])]))).collect()

#2
def rbk(*aes):
    l=[]
    for a in aes:
        if isinstance(a, list):
            l.extend(a)
        else:
            l.append(a)
    return sorted(l)

rdd.reduceByKey(rbk).map(lambda x:(x[0],','.join([str(i) for i in x[1]]))).collect()

#3
rdd=sc.textFile('hdfs://master:8020/secondary_sort/input/sample_input.txt')
def date_temp_f(s):
    s=s.split(',')
    return (s[0]+'-'+s[1],int(s[3])),int(s[3])

rdd=rdd.map(date_temp_f)
rdd.sortBy(keyfunc=lambda x:x[1]).groupBy(lambda x:x[0][0]).map(lambda x:(x[0],','.join([str(i[1]) for i in x[1]]))).collect()

#4
def append(a,b):
    a.append(b)
    return a

def extend(a,b):
    a.extend(b)
    return a

rdd.aggregateByKey(zeroValue=[],seqFunc=append,combFunc=extend).map(lambda x:(x[0],','.join([str(i) for i in sorted(x[1])]))).collect()

#5 可伸缩
rdd.sortBy(lambda x:x).groupBy(lambda x:x[0]).map(lambda x:(x[0],','.join([str(i[1]) for i in x[1]]))).collect()
