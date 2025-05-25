#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=02:00:00
#SBATCH --output=out/main_output_%j.log
#SBATCH --error=out/main_error_%j.log

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gmd_updated

# Print GPU info
nvidia-smi

# Run the script with Python's unbuffered output for real-time logging
PYTHONUNBUFFERED=1 python Aurel/skeleton_fitting/main_test.py 