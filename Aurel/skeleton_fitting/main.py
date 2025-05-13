import os
from utils.animation import run_animation_workflow
from utils.regress_joints import JointRegressor

# Data paths
npy_file = "Aurel/skeleton_fitting/dno_example_output_save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/npy_files/results.npy"

# Define kinematic chain
kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# Animate joint locations of DNO output
motion_data, joint_distances_df, distance_data = run_animation_workflow(
    npy_file=npy_file,
    kinematic_chain=kinematic_chain,
    fps=20
)

print(f"Processed {npy_file} successfully.")






# Create regressor
regressor = JointRegressor(output_dir="Aurel/skeleton_fitting/output")

# Regress SKEL joints (neutral gender)
regressor.regress_skel_joints(
    gender="female",
    create_visualizations=True,
    rotate_output=True,
    num_frames=10,
    data_rate=60.0
)

# Convert motion data from NPY to TRC
# regressor.convert_npy_to_trc(
#     npy_path="output/results.npy",
#     output_name="smpl_motion_jump",
#     data_rate=30.0,
#     rotate_output=False
# ) 