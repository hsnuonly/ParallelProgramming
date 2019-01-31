import subprocess
import csv
import re
import sys
import os

# version = sys.argv[1]
f = open('measurement.csv', 'w')
csv_writer = csv.writer(f)
csv_writer.writerow(['Processes', 'Computation', 'IO', 'Communication'])

N = [1, 1, 2, 4, 4]
n = [1, 2, 4, 8, 16]
c = [1, 1, 1, 1, 1]

for i in range(len(N)):
    os.environ['omp_threads'] = str(c[i])
    p = subprocess.Popen(['srun', '-x', 'apollo[34]', '-p', 'batch', '-n', str(n[i]), '-N', str(N[i]),
                          '-c', str(c[i]), 'ssspm', 'dense_3000.in', str(i) + 'out'], stdout=subprocess.PIPE)
    res = [i + 1]
    for line in p.stdout:
        if line != '':
            res += line.decode('utf8').split(' ')
    print(res)
    csv_writer.writerow(res)
f.close()
