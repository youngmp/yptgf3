#!/bin/bash


#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=park.y@ufl.edu   # Where to send mail
#SBATCH --ntasks=50 # number of cores to use
#SBATCH --time 23:00:00 # total time
#SBATCH --mem-per-cpu=8000mb # don't need much for this

pwd; hostname; date

module load python

python3 generate_figures.py

date
