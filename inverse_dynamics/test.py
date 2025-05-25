from inverse_dynamics.inverse_kinematics import run_ik
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
smpl_joints = np.load(npy_file, allow_pickle=True).item()['motion']

smpl_joints = smpl_joints[0].transpose(2,0,1)

smpl_joints = torch.from_numpy(smpl_joints).to(device)

output = run_ik(input_joints=smpl_joints, debug=False)

print(output["total_loss"])
print(output["translational_loss"])
print(output["rotational_loss"])

