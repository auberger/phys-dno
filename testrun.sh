#!/bin/bash
    
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/GRF.out

#nvidia-smi
python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --seed 42 --text_prompt "a person is doing a long jump" 
