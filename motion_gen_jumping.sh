#!/bin/bash
  
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/jumping_crazy.out

#nvidia-smi

text_prompt="a person is doing a jump"
seed=52

# normal - 1 iteration
#(0, 30, [1], [0.9686]),
#(0, 60, [1], [1.8800]),
#(0, 90, [1], [0.9290]) 

# long forward jump - 100 iterations then 800 iterations
#(0, 60, [2], [1.7658])

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"
