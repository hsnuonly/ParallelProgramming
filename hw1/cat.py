import sys
from array import array

f1 = open(sys.argv[1],'rb')
f2 = open(sys.argv[2],'rb')

a1 = array('f')
a1.fromstring(f1.read())

a2 = array('f')
a2.fromstring(f2.read())

for i in a1:
    print(i)