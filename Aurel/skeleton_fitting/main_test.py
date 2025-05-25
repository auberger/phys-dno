import os
import torch
import time
from utils.animation import run_animation_workflow, animate_with_contact_forces
import utils.anatomical_joint_regressor as joint_regressor
import utils.anatomical_joint_ik_adam as ik_fitter
import utils.contact_models_torch as contact_models_torch
from utils.center_of_mass_calculator import CenterOfMassCalculator
from utils.losses import calculate_dynamical_consistency_losses


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data path
npy_file = "save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/results.npy"

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








######################### Calculate Center of Mass and Inertia #########################
# Initialize the center of mass calculator
com_calculator = CenterOfMassCalculator()  

# Calculate center of mass properties and save as JSON
com_results = com_calculator.calculate_and_save_center_of_mass(
    joints=results['joints'], 
    joints_ori=results['joints_ori'],
    output_dir="Aurel/skeleton_fitting/output/com_analysis",
    filename="com_analysis_results.json",
    save_format="json",  # Save as JSON instead of PyTorch tensor
    save_results=True,  # Enable saving (can be set to False to skip saving)
    print_summary=True,  # Enable printing summary (can be set to False to skip printing)
    fps=20.0,  # Frame rate for accurate velocity and acceleration calculation
    smooth_derivatives=True  # Apply Savitzky-Golay smoothing for better derivatives (CoM velocity and acceleration)
)


######################### Run contact model to get ground reaction forces and torques #########################
# Initialize model
contact_model = contact_models_torch.ContactModel()

# Calculate contact forces and CoP
output = contact_model(results["joints"], results["joints_ori"])

######################### Calculate Dynamical Consistency Losses #########################

# Calculate basic loss (scalar value)
total_loss = calculate_dynamical_consistency_losses(
    com_results=com_results,
    contact_output=output,
    translational_weight=1.0,
    rotational_weight=1.0,
    return_detailed=False, # set to True to get detailed breakdown of losses
    print_summary=True
)