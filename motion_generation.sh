#!/bin/bash
  
#SBATCH --error=out/running.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/running.out

#nvidia-smi

text_prompt="a person is running"
num_opt_steps=800
seed=10

echo "=== Motion Generation Job ==="
echo "NORMAL"
echo "Text Prompt       : $text_prompt"
echo "Optimization Steps: $num_opt_steps"
echo "Seed              : $seed"
echo
echo "Target edit list:"
cat <<EOF
      (0, 30, [0], [0.0026]),
      (0, 30, [2], [0.0044]),

      (0, 60, [0], [0.2826]),  # go right a bit
      (0, 60, [2], [0.3888]),

      (0, 90, [2], [1.6109]),  # go right a bit
      (0, 90, [2], [1.8909])
EOF
echo


python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"

