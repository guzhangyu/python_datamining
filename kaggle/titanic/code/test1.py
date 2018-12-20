# import pandas as pd
#
# a=pd.DataFrame({'Name':['ss'],'P':['de']})
# print(a['Name'].isnan())
# print(a['Name'].str.contains('s'))
#
# exit()

# import re
#
# family_name=re.compile(', (?:Dr|M(?:aster|rs|r|iss|s))\. \(?(\w+)')
# print(re.search(family_name,'Sage, Mr. Douglas Bullen ').groups())
# print(re.search(family_name,'Sage, Dr. Douglas Bullen ').groups())
# print(re.search(family_name,'O\'Donoghue, Ms. Bridget').groups())
# family_name.match('Sage, Mr. Douglas Bullen ').group(1)
from types import FunctionType


def a(b,*a) -> int:
    print(b)
    print(a)
    return 's'

# e={'a':1,'b':2}.items()
# a(e)
print(a(1,({'3':2},4)))
print(type(1))
print(a.__class__ ==FunctionType)
print(a.__code__.__class__)