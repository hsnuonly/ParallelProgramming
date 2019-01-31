import subprocess
import csv
import re
import sys

version = sys.argv[1]
f = open('measurement.csv','w')
csv_writer = csv.writer(f)
csv_writer.writerow(['Processes','Computation','IO','Communication'])

for i in range(4):
    p = subprocess.Popen(['srun','-p','batch','-N',str(i+1),'-n',str(1<<i),version,str(i<<i),'-2','2','-2','2','1000','1000','test.png'],stdout=subprocess.PIPE)
    res = [i+1]
    for line in p.stdout:
        if line != '':
            res+=line.decode('utf8').split(' ')
    print(res)
    csv_writer.writerow(res)
f.close()
