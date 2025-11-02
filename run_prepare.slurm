#!/bin/bash
#SBATCH --account=course_cap6614
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --error=slurm_outputs/prepare.err
#SBATCH --output=slurm_outputs/prepare.out
#SBATCH --job-name=prepare

module load cuda/cuda-12.2.2
module load python/python-3.11.4-gcc-12.2.0

source ~/Venvs/MARINE/bin/activate
cd ~/Projects/CAP6614_MARINE/
bash scripts/eval_instructBLIP.sh