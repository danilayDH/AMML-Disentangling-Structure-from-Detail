#!/bin/bash -eux

#SBATCH --job-name=downstream
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=jacob.schaefer@student.hpi.de
#SBATCH --partition=gpupro
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH -o slogs/%x_%A_%a.log


srun python src/downstream_module.py "$@"

