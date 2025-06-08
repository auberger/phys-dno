#!/bin/bash
  
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/walking.out

#nvidia-smi

text_prompt="a person is walking"
seed=13

# normal - 1 iteration
# (0, 30, [2], [0.0044]), 
# (0, 60, [2], [0.3888]), 
# (0, 90, [2], [1.8909])

# walking turn left, left arc - 100 iterations then 800 iterations
# (0, 20, [0], [-0.2192]), 
# (0, 20, [2], [0.8807]), 
# (0, 40, [0], [-1.1151]),
# (0, 40, [2], [1.5184])

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"