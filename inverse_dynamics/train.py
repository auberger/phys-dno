import inverse_dynamics.utils.anatomical_joint_regressor as ajr
import torch
import numpy as np

output_dir = "./inverse_dynamics/regressor"
gender = "male"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
smpl_joints = np.load(npy_file, allow_pickle=True).item()['motion']
smpl_joints = smpl_joints[0].transpose(2,0,1)
smpl_joints = torch.from_numpy(smpl_joints).to(device)

regressor = ajr.SparseSMPLtoAnatomicalRegressor(output_dir=output_dir, gender=gender, debug=True)
regressor.run_pipeline(num_samples=20000, alpha=0.0005, learning_rate=0.005, num_epochs=2000)
pred_joints = regressor.predict_anatomical_joints(smpl_joints, output_dir)

print(pred_joints[0][19])