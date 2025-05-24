from inverse_dynamics.inverse_kinematics import run_ik
import numpy as np

npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
smpl_joints = np.load(npy_file, allow_pickle=True).item()['motion']

smpl_joints = smpl_joints[0].transpose(2,0,1)

output = run_ik(input_joints=smpl_joints, debug=True)

print(output['total_force'])