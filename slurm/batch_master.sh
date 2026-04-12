#!/bin/bash

#SBATCH --job-name=master   # Job name
#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=park.y@ufl.edu   # Where to send mail
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --output=log/master_B5.1_ze0.6_%A-%a.log
#SBATCH --mem=1gb

pwd; hostname; date

module load python

python3 master.py -s $SLURM_ARRAY_TASK_ID -B 5.1 -z 0.6
#echo This is task $SLURM_ARRAY_TASK_ID

date
