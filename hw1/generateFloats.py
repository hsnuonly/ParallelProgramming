import random
import os
from array import array

try:
    os.remove('testcase')
except:
    pass
a = array('f')

for _ in range(100):
    a.append(random.randint(0,10000000))

f = open('testcase','wb+')
a.tofile(f)
f.close()
