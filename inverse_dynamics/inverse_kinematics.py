import os
import torch
import time
from inverse_dynamics.utils.animation import run_animation_workflow, animate_with_contact_forces
import inverse_dynamics.utils.anatomical_joint_regressor as joint_regressor
import inverse_dynamics.utils.anatomical_joint_ik_adam as ik_fitter
import inverse_dynamics.utils.contact_models_torch as contact_models_torch
from inverse_dynamics.utils.center_of_mass_calculator import CenterOfMassCalculator
from inverse_dynamics.utils.losses import calculate_dynamical_consistency_losses


def run_ik(input_joints=None, npy_file="", debug=False, initial_poses=None, initial_trans=None):

    if (input_joints is None and npy_file == ""): raise RuntimeError("inverse kinematics requires input joints or an input file")

    if (input_joints is not None): mode = 'JOINT_MODE'
    else: mode = 'FILE_MODE'

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if debug: print(f"Using device: {device}")

    # Animate joint locations of DNO output
    #motion_data, joint_distances_df, distance_data = run_animation_workflow(npy_file=npy_file)

    ######################### Apply regressor to motion sequence to get anatomical joint locations from SMPL joints #########################
    if (mode == 'FILE_MODE'):
        regressor_dir = "Aurel/skeleton_fitting/output/regressor"
        output_dir = "Aurel/skeleton_fitting/output/regressor"
        output_file = os.path.join(output_dir, "regressed_joints.npy")
    if (mode == 'JOINT_MODE'):
        regressor_dir = "inverse_dynamics/regressor"
        output_dir = "inverse_dynamics/output"

    trial_idx = 0

    # Time the regressor initialization and prediction
    start_time = time.time()
    if debug: print("Initializing regressor...")
    regressor = joint_regressor.SparseSMPLtoAnatomicalRegressor(output_dir=output_dir, debug=debug) # go to utils/anatomical_joint_regressor.py to change or retrain the regressor
    init_time = time.time() - start_time
    if debug: print(f"Regressor initialization took {init_time:.2f} seconds")

    if debug: print("Running regressor prediction...")
    start_time = time.time()
    if (mode == 'FILE_MODE'):
        anatomical_joints = regressor.predict_anatomical_joints_from_file(
            npy_file, 
            regressor_dir, 
            output_file=output_file,
            trial=trial_idx
        )
    if (mode == 'JOINT_MODE'):
        anatomical_joints = regressor.predict_anatomical_joints(
            input_joints, 
            regressor_dir, 
            trial=trial_idx
        )
    regression_time = time.time() - start_time
    if debug: print(f"Regressor prediction took {regression_time:.2f} seconds")

    ######################### Run IK to get anatomical joint locations from anatomical joint locations #########################
    if debug: print("Initializing IK fitter...")
    start_time = time.time()
    fitter = ik_fitter.AnatomicalJointFitter(
        output_dir=output_dir,
        debug=debug,
        device=device,  # Pass the device to the fitter
        initial_poses=initial_poses,
        initial_trans=initial_trans
    )
    fitter_init_time = time.time() - start_time
    if debug: print(f"IK fitter initialization took {fitter_init_time:.2f} seconds")

    if debug: print("Running IK optimization...")
    start_time = time.time()
    if (mode == 'FILE_MODE'):
        fitting_results = fitter.run_ik_from_file(
            anatomical_joints_file=output_file,
            output_file="jumping_ik_results.pkl",
            max_iterations=150,
            learning_rate=0.1,
            pose_regularization=0.001,
            trial=trial_idx
        )
    if (mode == 'JOINT_MODE'):
        fitting_results = fitter.run_ik(
            anatomical_joints=anatomical_joints,
            max_iterations=150,
            learning_rate=0.1,
            pose_regularization=0.001,
            trial=trial_idx
        )
    ik_time = time.time() - start_time
    if debug: print(f"IK optimization took {ik_time:.2f} seconds")


    ######################### Calculate Center of Mass and Inertia #########################
    # Initialize the center of mass calculator
    com_calculator = CenterOfMassCalculator()  

    # Calculate center of mass properties and save as JSON
    com_results = com_calculator.calculate_and_save_center_of_mass(
        joints=fitting_results['joints'], 
        joints_ori=fitting_results['joints_ori'],
        save_results=False,  # Enable saving (can be set to False to skip saving)
        print_summary=debug,  # Enable printing summary (can be set to False to skip printing)
        fps=20.0,  # Frame rate for accurate velocity and acceleration calculation
        smooth_derivatives=True  # Apply Savitzky-Golay smoothing for better derivatives (CoM velocity and acceleration)
    )


    ######################### Run contact model to get ground reaction forces and torques #########################
    # Initialize model
    contact_model = contact_models_torch.ContactModel()

    # Calculate contact forces and CoP
    contact_output = contact_model(fitting_results["joints"], fitting_results["joints_ori"])

    # Calculate basic loss (scalar value)
    total_loss = calculate_dynamical_consistency_losses(
        com_results=com_results,
        contact_output=contact_output,
        translational_weight=1.0,
        rotational_weight=1.0,
        return_detailed=True, # set to True to get detailed breakdown of losses
        print_summary=debug
    )
    """{
        "total_loss": total_loss,
        "translational_loss": translational_loss,
        "rotational_loss": rotational_loss,
        "force_residual": force_residual,
        "moment_residual": moment_residual,
        "required_force": required_force,
        "grf_moment_about_com": grf_moment_about_com,
        "valid_cop_frames": valid_cop_mask.sum().item(),
        "total_frames": B
    }"""
    """
            fitting_results['joints']
            fitting_results['joints_ori']
            fitting_results['poses']
            fitting_results['trans']
            fitting_results['betas']
            fitting_results['joint_errors']
    """
    return total_loss, contact_output, fitting_results

if __name__ == "__main__":
    npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy"
    run_ik(npy_file=npy_file, debug=True)