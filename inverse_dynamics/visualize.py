import os
import numpy as np
import sys

# Set current working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of this script
project_root = os.path.dirname(script_dir)  # Go up one level to project root
os.chdir(project_root)  # Change to project root
print(f"Current working directory set to: {os.getcwd()}")

# Add project root to Python path for imports
sys.path.insert(0, project_root)

from inverse_dynamics.utils.animation import animate_with_dynamics_analysis, animate_with_angular_dynamics_analysis
import inverse_dynamics.utils.anatomical_joint_regressor as joint_regressor
import inverse_dynamics.utils.anatomical_joint_ik_adam as ik_fitter
import torch

# Add the inverse dynamics path to import modules
import inverse_dynamics.utils.contact_models_torch as contact_models_torch
from inverse_dynamics.utils.center_of_mass_calculator import CenterOfMassCalculator
from inverse_dynamics.utils.losses import calculate_dynamical_consistency_losses


######################### User Input #########################
# Data path
npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
output_dir = "inverse_dynamics/output"  # Relative path from project root
trial_idx = 1

######################### Apply regressor to motion sequence to get anatomical joint locations from SMPL joints #########################
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_file = os.path.join(output_dir, "regressed_joints.npy")

# Load the regressor
regressor = joint_regressor.SparseSMPLtoAnatomicalRegressor(output_dir=output_dir) # go to utils/anatomical_joint_regressor.py to change or retrain the regressor
anatomical_joints = regressor.predict_anatomical_joints(
    npy_file, 
    "inverse_dynamics/regressor",
    gender="male",
    output_file=output_file,
    trial=trial_idx
)

######################### Run IK to get anatomical joint locations from anatomical joint locations #########################
fitter = ik_fitter.AnatomicalJointFitter(
    output_dir=output_dir,
    debug=True
)

results = fitter.run_ik(
    anatomical_joints=anatomical_joints,
    output_file="jumping_ik_results.pkl",
    max_iterations=150,
    learning_rate=0.1,
    pose_regularization=0.001,
    trial=trial_idx
)

######################### Calculate Center of Mass and Moment #########################
# Initialize the center of mass calculator
com_calculator = CenterOfMassCalculator()  

# Calculate center of mass properties including angular momentum and moment
com_results = com_calculator.calculate_and_save_center_of_mass(
    joints=results["joints"], 
    joints_ori=results["joints_ori"],
    output_dir=output_dir,
    filename="com_analysis_results.json",
    save_format="json",  # Save as JSON instead of PyTorch tensor
    save_results=True,  # Enable saving (can be set to False to skip saving)
    print_summary=True,  # Enable printing summary (can be set to False to skip printing)
    fps=20.0,  # Frame rate for accurate velocity and acceleration calculation
    smooth_derivatives=True  # Apply Savitzky-Golay smoothing for better derivatives
)

######################### Run contact model to get ground reaction forces and torques #########################
# Initialize model
contact_model = contact_models_torch.ContactModel()

# Calculate contact forces and CoP
output = contact_model(results["joints"], results["joints_ori"])

# Save contact model output for visualization
contact_output_file = os.path.join(output_dir, "contact_forces/contact_output.pt")
os.makedirs(os.path.dirname(contact_output_file), exist_ok=True)
torch.save({
    'sphere_positions': output.sphere_positions,
    'sphere_forces': output.sphere_forces,  # Save sphere-specific forces
    'force': output.force,
    'force_right': output.force_right,
    'force_left': output.force_left,
    'cop': output.cop,
    'cop_right': output.cop_right,
    'cop_left': output.cop_left
}, contact_output_file)

######################### Calculate Dynamical Consistency Losses #########################

# Calculate basic loss (scalar value)
total_loss = calculate_dynamical_consistency_losses(
    com_results=com_results,
    contact_output=output,
    fps=20.0,
    gravity=9.81,
    translational_weight=1.0,
    rotational_weight=1.0,
    return_detailed=False, # set to True to get detailed breakdown of losses
    print_summary=True
)

######################### Create Enhanced Animation with Linear Dynamics Analysis #########################

# Create animation with dynamics analysis
ani = animate_with_dynamics_analysis(
    joints=results["joints"],
    joints_ori=results["joints_ori"],
    trans=results["trans"],
    contact_output=output,
    com_results=com_results,  # Pass the COM analysis results
    fps=20,  # Adjust frame rate as needed
    force_scale=0.001,  # Adjust to make force vectors visible (smaller = larger vectors)
    save_path=os.path.join(output_dir, "linear_dynamics_animation.gif")  # Save as GIF
)

######################### Create Enhanced Animation with Angular Dynamics Analysis #########################

# Create animation with angular dynamics analysis
ani_angular = animate_with_angular_dynamics_analysis(
    joints=results["joints"],
    joints_ori=results["joints_ori"],
    trans=results["trans"],
    contact_output=output,
    com_results=com_results,  # Pass the COM analysis results
    fps=20,  # Adjust frame rate as needed
    moment_scale=0.01,  # Adjust to make moment vectors visible (smaller = larger vectors)
    save_path=os.path.join(output_dir, "angular_dynamics_animation.gif")  # Save as GIF
)