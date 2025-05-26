#!/bin/bash
  
#SBATCH --error=out/running.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/running.out

#nvidia-smi

text_prompt="a person is running"
seed=52

# normal - 1 iteration
# (0, 30, [2], [2.4151]),  
# (0, 60, [2], [2.8609]), 
# (0, 90, [2], [3.6772]), 

# running turn left, left arc - 100 iterations then 800 iterations
# ???

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"