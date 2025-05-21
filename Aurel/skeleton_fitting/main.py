import os
from utils.animation import run_animation_workflow
import utils.anatomical_joint_regressor as joint_regressor
import utils.anatomical_joint_ik_adam as ik_fitter

# Data path
npy_file = "Aurel/skeleton_fitting/dno_example_output_save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/results.npy"

# Animate joint locations of DNO output
motion_data, joint_distances_df, distance_data = run_animation_workflow(npy_file=npy_file)

######################### Apply regressor to motion sequence to get anatomical joint locations from SMPL joints #########################
output_dir = "Aurel/skeleton_fitting/output/regressor"
output_file = os.path.join(output_dir, "regressed_joints.npy")
trial_idx = 1

regressor = joint_regressor.SparseSMPLtoAnatomicalRegressor(output_dir=output_dir) # go to utils/anatomical_joint_regressor.py to change or retrain the regressor
anatomical_joints = regressor.predict_anatomical_joints(
    npy_file, 
    output_dir, 
    output_file=output_file,
    trial=trial_idx
)

######################### Run IK to get anatomical joint locations from anatomical joint locations #########################
fitter = ik_fitter.AnatomicalJointFitter(
    output_dir="Aurel/skeleton_fitting/output/ik_fitter",
    debug=True
)

results = fitter.run_ik(
    anatomical_joints_file=output_file,
    output_file="jumping_ik_results.pkl",
    max_iterations=150,
    learning_rate=0.1,
    pose_regularization=0.001,
    trial=trial_idx
)

# Visulaze IK results by running the following in the command line (adapt the path to the results file):
# python external/aitviewer-skel/examples/load_SKEL.py -s '/Users/auberger/Documents/Github_repos/phys-dno/Aurel/skeleton_fitting/output/ik_fitter/jumping_ik_results.pkl'