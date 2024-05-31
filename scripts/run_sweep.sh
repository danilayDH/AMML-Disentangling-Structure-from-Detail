#!/bin/bash -eux

#SBATCH --job-name=sweep_no_mask
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=daniela.layer@student.hpi.de
#SBATCH --partition=gpupro,gpua100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH -o slogs/%x_%A_%a.log


# pass the sweep's ID as the first argument, e.g., sbatch this_script.sh wvsz8nl6
# this will only run a single configuration from your sweep
# a convenient way to run multiple configurations in parallel is with the use of slurm arrays, e.g.:
# sbatch --array=0-N%M run_sweep.sh sweep_id
# where N is the total number of configurations you want to run, and M is the maximum number of jobs that would run in parallel
wandb agent --count 1 "aml_ws2324/Brain Disentanglement Dementia/${1}"
