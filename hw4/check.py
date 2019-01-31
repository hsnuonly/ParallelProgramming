import sys
from array import array

f1 = open(sys.argv[1],'rb')
f2 = open(sys.argv[2],'rb')

a1 = array('i')
a1.fromstring(f1.read())

a2 = array('i')
a2.fromstring(f2.read())

count = 0
wrong = 0

for i in zip(a1,a2):
    if i[0]!=i[1]:
        print('%d %8d %8d%2s'%(count,i[0],i[1],'*' if i[0]!=i[1] else ''))
        wrong += 1
    count+=1

print('%d/%d'%(wrong,count))
 
f1.close()
f2.close()