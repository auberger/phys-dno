#!/bin/bash

#SBATCH --output=testrun_%j.out    
#SBATCH --error=testrun_%j.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00

#nvidia-smi

python -m eval.eval_refinement --model_path ./save/model000500000_avg.pt