#!/bin/bash
  
#SBATCH --error=out/walking.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/walking.out

#nvidia-smi

text_prompt="a person is waliking"
seed=52

# normal - 1 iteration
# (0, 30, [2], [0.0044]), 
# (0, 60, [2], [0.3888]), 
# (0, 90, [2], [1.8909])

# walking turn left, left arc - 100 iterations then 800 iterations
# ???

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"