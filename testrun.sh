#!/bin/bash

#SBATCH --output=testrun_%j.out    
#SBATCH --error=testrun_%j.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/testtest.out

#nvidia-smi
python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "a person is walking"
