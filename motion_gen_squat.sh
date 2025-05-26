#!/bin/bash
  
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/squat_last.out

#nvidia-smi

text_prompt="a person is doing a squat"
seed=52

# normal - 1 iteration
# ??

# ankles 20cm (0.2) wider in x axis - 100 iterations then 800 iterations
# seed 52_ - 300 iter
# (0, 10, [1], [0.8983]),
# (0, 60, [1], [0.3456]),
# (4, 60, [0], [0.2922]),  
# (5, 60, [0], [-0.3035])

        #(0, 10, [1], [0.8983]),
        #(0, 60, [1], [0.3456]),
        #(4, 20, [0], [0.2922]),  
        #(5, 20, [0], [-0.3035]),
        #(4, 40, [0], [0.2922]),  
        #(5, 40, [0], [-0.3035]),
        #(4, 60, [0], [0.2922]),  
        #(5, 60, [0], [-0.3035]),
        #(4, 100, [0], [0.2922]),  
        #(5, 100, [0], [-0.3035]),

python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "$text_prompt" --seed "$seed"
