#!/bin/bash -eux

#SBATCH --job-name=test_vae
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=daniela.layer@student.hpi.de
#SBATCH --partition=gpupro
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH -o slogs/%x_%A_%a.log


srun python src/test_vae.py "$@"

