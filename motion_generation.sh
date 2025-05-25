#!/bin/bash
  
#SBATCH --error=out/standsup.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/standsup.out

#nvidia-smi

text_prompt="a person stands up"
num_opt_steps=100

echo "Generating motion: '$text_prompt'"
echo "Number of opt steps: '$num_opt_steps'"

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed 10

