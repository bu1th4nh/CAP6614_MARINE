#!/bin/bash
#SBATCH --account=course_cap6614
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --error=blip_latest.err
#SBATCH --output=blip_latest.out
#SBATCH --job-name=cap6614_bu1th4nh_instructBLIP


set -euo pipefail

# Minimal, predictable environment
module purge
module load cuda/cuda-12.2.2
module load python/python-3.11.4-gcc-12.2.0


if ssh -fN HEC-301-14-Server; then
    echo "SSH tunnel established successfully."
else
    echo "Failed to establish SSH tunnel; continuing without it." 
fi

source ~/Venvs/MARINE/bin/activate
cd ~/Projects/CAP6614_MARINE/

set -a
source ./.env
set +a

srun bash scripts/eval_instructBLIP.sh