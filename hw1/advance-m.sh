#!/bin/bash
#SBATCH -n 4
#SBATCH -N 2
#SBATCH -p batch
#SBATCH -W
#STATCH -o time.txt
srun ./advanced-measurement 40000000 30.in out.txt

