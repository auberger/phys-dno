#!/bin/bash
  
#SBATCH --error=out/squat.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/squat.out

#nvidia-smi

text_prompt="a person is doing a squat"
seed=52

# normal - 1 iteration
# ??

# ankles 20cm (0.2) wider in x axis - 100 iterations then 800 iterations
# ??

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"
