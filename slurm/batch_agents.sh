#!/bin/bash

###
## PLEASE USE slurm_agents.sh TO SUBMIT JOBS, NOT THIS FILE.
#

#SBATCH --mail-type=FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=park.y@ufl.edu   # Where to send mail
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --mem=7gb
#SBATCH --array=0-49                 # Array range
#SBATCH -t 4-00:00

pwd; hostname; date

module load python

python3 agents.py -s $SLURM_ARRAY_TASK_ID -B ${B} -T ${T} -z ${z} --dt 1e-5 --save_hist --save_switch
#echo This is task $SLURM_ARRAY_TASK_ID

date
