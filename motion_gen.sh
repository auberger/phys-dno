#!/bin/bash
  
#SBATCH --error=out/squat.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/squat.out

#nvidia-smi

text_prompt="a person is squatting"
seed=52

echo "=== Motion Generation Job ==="
echo "NORMAL"
echo "Text Prompt       : $text_prompt"
echo "Seed              : $seed"

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"
