import re

a=re.compile('a\t')
for i in a.finditer("a	"):
    print(i.groupdict())
print(a.sub("","a b"))
