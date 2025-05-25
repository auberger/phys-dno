#!/bin/bash

#SBATCH --output=testrun_%j.out    
#SBATCH --error=testrun_%j.err     
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output out/walking.out

#nvidia-smi

echo "Generating motion: 'a person is walking'"
echo "target_edit_list = [
	(0, 40, [0], -1.2),
	(0, 80, [0], 2.5),
    ]
    kframes = []
    obs_list = [
        ((-0.5, 3.5) , 0.7)
    ]"
python -m sample.gen_dno --model_path ./save/model000500000_avg.pt --text_prompt "a person is walking"
