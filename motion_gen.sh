#!/bin/bash
  
#SBATCH --error=out/squat.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/squat.out

#nvidia-smi

text_prompt="a person squats"
num_opt_steps=800
seed=42

echo "=== Motion Generation Job ==="
echo "NORMAL"
echo "Text Prompt       : $text_prompt"
echo "Optimization Steps: $num_opt_steps"
echo "Seed              : $seed"

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"
