#!/bin/bash

#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=00:00:10
#SBATCH --output out/avoiding.out

#nvidia-smi
python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "a person is avoiding a box"
