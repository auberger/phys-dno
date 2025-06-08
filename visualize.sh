#!/bin/bash
  
#SBATCH --error=out/visualize.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/visualize.out

#nvidia-smi

python -m inverse_dynamics.visualize
