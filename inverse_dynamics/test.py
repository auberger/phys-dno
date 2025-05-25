from inverse_dynamics.inverse_kinematics import run_ik
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
smpl_joints = np.load(npy_file, allow_pickle=True).item()['motion']

smpl_joints = smpl_joints[0].transpose(2,0,1)

smpl_joints = torch.from_numpy(smpl_joints).to(device)

output = run_ik(input_joints=smpl_joints, debug=True)

#print(output['total_force'][100])
grf = output['total_force']
grf_magnitude = torch.norm(grf, p=2, dim=1)
gravity = torch.tensor(735.75, device=device) # 75kg * 9.81 ms^-2
target_forces = torch.full_like(grf_magnitude, gravity)
print(f"First 5 magnitudes:\n{grf_magnitude[:5].detach().cpu().numpy()}")

loss_fn = nn.MSELoss()

magnitude_loss = loss_fn(grf_magnitude, target_forces)

print(f"Calculated Magnitude MSE Loss: {magnitude_loss.item()}")

