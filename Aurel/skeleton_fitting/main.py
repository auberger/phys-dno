import os
import torch
import time
from utils.animation import run_animation_workflow, animate_with_contact_forces
import utils.anatomical_joint_regressor as joint_regressor
import utils.anatomical_joint_ik_adam as ik_fitter
import Aurel.skeleton_fitting.utils.contact_models_torch as contact_models_torch
import matplotlib
matplotlib.use("TkAgg")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data path
npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"

# Animate joint locations of DNO output
#motion_data, joint_distances_df, distance_data = run_animation_workflow(npy_file=npy_file)

######################### Apply regressor to motion sequence to get anatomical joint locations from SMPL joints #########################
output_dir = "Aurel/skeleton_fitting/output/regressor"
output_file = os.path.join(output_dir, "regressed_joints.npy")
trial_idx = 0

# Time the regressor initialization and prediction
start_time = time.time()
print("Initializing regressor...")
regressor = joint_regressor.SparseSMPLtoAnatomicalRegressor(output_dir=output_dir) # go to utils/anatomical_joint_regressor.py to change or retrain the regressor
init_time = time.time() - start_time
print(f"Regressor initialization took {init_time:.2f} seconds")

print("Running regressor prediction...")
start_time = time.time()
anatomical_joints = regressor.predict_anatomical_joints(
    npy_file, 
    output_dir, 
    output_file=output_file,
    trial=trial_idx
)
regression_time = time.time() - start_time
print(f"Regressor prediction took {regression_time:.2f} seconds")

######################### Run IK to get anatomical joint locations from anatomical joint locations #########################
print("Initializing IK fitter...")
start_time = time.time()
fitter = ik_fitter.AnatomicalJointFitter(
    output_dir="Aurel/skeleton_fitting/output/ik_fitter",
    debug=True,
    device=device  # Pass the device to the fitter
)
fitter_init_time = time.time() - start_time
print(f"IK fitter initialization took {fitter_init_time:.2f} seconds")

print("Running IK optimization...")
start_time = time.time()
results = fitter.run_ik(
    anatomical_joints_file=output_file,
    output_file="jumping_ik_results.pkl",
    max_iterations=150,
    learning_rate=0.1,
    pose_regularization=0.001,
    trial=trial_idx
)
ik_time = time.time() - start_time
print(f"IK optimization took {ik_time:.2f} seconds")

######################### Run contact model to get ground reaction forces and torques #########################
# Initialize model and move it to the correct device
print("Initializing contact model...")
start_time = time.time()
contact_model = contact_models_torch.ContactModel().to(device)
contact_init_time = time.time() - start_time
print(f"Contact model initialization took {contact_init_time:.2f} seconds")

# Move input tensors to the correct device
print("Computing contact forces...")
start_time = time.time()
joints = results['joints'].to(device)
joints_ori = results['joints_ori'].to(device)

# Calculate contact forces and CoP
output = contact_model(joints, joints_ori)
contact_time = time.time() - start_time
print(f"Contact force computation took {contact_time:.2f} seconds")

# Total forces and torques
total_force = output.force
total_cop = output.cop

# Right foot forces and torques
right_force = output.force_right
right_cop = output.cop_right

# Left foot forces and torques
left_force = output.force_left
left_cop = output.cop_left

# Sphere positions
sphere_positions = output.sphere_positions

# Print total execution time
total_time = init_time + regression_time + fitter_init_time + ik_time + contact_init_time + contact_time
print("\nTotal execution time breakdown:")
print(f"Regressor initialization: {init_time:.2f} seconds")
print(f"Regressor prediction: {regression_time:.2f} seconds")
print(f"IK fitter initialization: {fitter_init_time:.2f} seconds")
print(f"IK optimization: {ik_time:.2f} seconds")
print(f"Contact model initialization: {contact_init_time:.2f} seconds")
print(f"Contact force computation: {contact_time:.2f} seconds")
print(f"Total execution time: {total_time:.2f} seconds")

# Create animation
# ani = animate_with_contact_forces(
#     joints=results['joints'],
#     joints_ori=results['joints_ori'],
#     trans=results['trans'],  # Keep trans for animation since it might be needed for visualization
#     contact_output=output,
#     fps=20,  # Adjust frame rate as needed
#     force_scale=0.001  # Adjust to make forces more/less visible
# )