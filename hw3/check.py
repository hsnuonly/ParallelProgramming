import sys
from array import array

f1 = open(sys.argv[1], 'rb')
f2 = open(sys.argv[2], 'rb')

a1 = array('i')
a1.fromstring(f1.read())

a2 = array('i')
a2.fromstring(f2.read())

diff = 0

for i in zip(a1, a2):
    if(i[0] != i[1]):
        diff += 1
print(diff)
