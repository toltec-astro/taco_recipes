#!/bin/bash

# ----------------------------------------------------------------
# An SLURM job script template.
# 
# To learn more about SBATCH options,
# visit https://slurm.schedmd.com/sbatch.html 
# ----------------------------------------------------------------

#SBATCH -J reduce_sweep
#SBATCH -o %j-%x.out  # %j = job ID
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH --nodes=1  # Node count required for the job 
#SBATCH --ntasks=4  # Number of tasks to be launched 
#SBATCH --cpus-per-task=6  # Number of cores per task
#SBATCH --mem=16G  # Mem required per node
#SBATCH --partition toltec-cpu  # Partition
#SBATCH --parsable

if [[ ! $2 ]]; then
    echo "Usage: $0 <obsnum_start> <obsnum_end>"
    exit 1
fi
for (( i=$2; i>=$1; i-=1 )); do
    srun bash reduce.sh $i 
done
