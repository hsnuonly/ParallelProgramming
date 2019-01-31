from array import array
import sys

f = open(sys.argv[1],'rb')
a = array('f')
a.fromstring(f.read())

is_sorted = True
prev = None
count = 0

for i in a:
#    print(i)
    if prev is not None:
        is_sorted = is_sorted and i >= prev
        if prev>i:
            count += 1
            print(i)
            print("===========")
    prev = i

print ('is_sorted = ',is_sorted)
print(count)
