import re
pattern=re.compile('\s+.+')

with open('data_description.txt', 'r') as f:
    l=f.read()
    l=l.split('======')
    print(l)
    for ll in l:
        ll=ll.strip()
        ln=ll.split('\n')
        sv=ln[0][:ln[0].index(':')+1]
        print(",    '%s':{" % sv[:-1])
        k=0
        for i in range(1,len(ln)):
            s=ln[i].strip()
            if not s:
                k=k+1
                continue
            v = re.sub(pattern, '', s).replace('\n', '')
            print("        '%s':%d," % (v,i-k))
        print('    }')
    # for i in range(0,len(l)):
    #     v=re.sub(pattern,'',l[i]).replace('\n','')
    #     print("'%s':%d," % (v,len(l)-1-i+1))