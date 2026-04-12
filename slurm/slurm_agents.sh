#!/bin/bash

# master file for submitting multiple jobs

#sleep 4500

T=10

count=1
# for each B
#for i in $(seq 5.11 .01 5.2)
#for i in $(seq 5.01 .01 5.1)
#for i in $(seq 5.12 .01 5.13)
#for i in $(seq 5.02 .01 5.11)

arrayname=( 5.02 5.05 5.1 )
#arrayname=( 5.01 5.05 5.1 )
#arrayname=( 5.01 )
for i in "${arrayname[@]}"

do
    #for j in $(seq 1.6 .1 5)
    for j in $(seq 0.0 .1 15)
    do
	echo $i $j
	sbatch --job-name=${i}_${j}_${T} --output=log/agents_B${i}_ze${j}_T${T}.log --export=B=$i,z=$j,T=$T batch_agents.sh
	sleep 1 # pause to be kind to the scheduler
	if !(($count % 10)); then
	    sleep 1300
	fi
	((count++))
	
    done
done
