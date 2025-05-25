"""
This script provides inverse kinematics functionality to fit the SKEL model to anatomical joint centers.

The core functionality is adapted from the SKEL aligner.py:
https://github.com/ipsavitsky/SKEL/blob/main/skel/alignment/aligner.py

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Original authors: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.

Adaptation for anatomical joint fitting: [Your Name]
"""

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm
import math
from skel.skel_model import SKEL
import matplotlib.pyplot as plt
import skel.kin_skel as kin_skel

class AnatomicalJointFitter:
    """
    Fits SKEL model parameters to anatomical joint positions using inverse kinematics.
    Unlike the original SKEL aligner, this only optimizes for joint positions (not mesh vertices).
    """
    
    def __init__(self, 
                 gender="male", 
                 device=None, 
                 output_dir="output/fitter",
                 debug=False):
        """
        Initialize the fitter.
        
        Args:
            gender: Gender for the SKEL model ('male' or 'female')
            device: Device to run computations on (will use CUDA if available)
            output_dir: Directory to save output files
            debug: Enable debug visualization
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.gender = gender
        self.output_dir = output_dir
        self.debug = debug
        
        os.makedirs(output_dir, exist_ok=True)
        if self.debug: 
            print(f"Using device: {self.device}")
            print(f"Using gender: {self.gender}")
        
        # Load the SKEL model
        self.skel = SKEL(gender=gender).to(self.device)
        
        # We'll optimize only pose parameters (theta), keeping beta at 0
        self.num_pose_params = self.skel.num_q_params
        if self.debug: print(f"SKEL model has {self.num_pose_params} pose parameters")
        
        # Create a mapping between joint names and indices for easier reference
        self.joint_names = kin_skel.skel_joints_name
        self.joint_indices = {name: i for i, name in enumerate(self.joint_names)}
        
        # Define joint importance weights
        # We give higher weight to extremities and important joints
        self.joint_weights = torch.ones(len(self.joint_names), 3, device=self.device)
        
        # Higher weights for extremities and important joints
        important_joints = ["pelvis", "thorax", "head", "hand_r", "hand_l", 
                          "ankle_r", "ankle_l", "toes_r", "toes_l"]
        
        for joint in important_joints:
            if joint in self.joint_indices:
                self.joint_weights[self.joint_indices[joint]] = 2.0
        
        # Pelvis is the root, so give it the highest weight
        self.joint_weights[self.joint_indices["pelvis"]] = 3.0
        
        # Add extra weights for problematic joints
        problematic_joints = {
            "hand_r": 3.0,
            "hand_l": 3.0,
            "humerus_r": 2.5,
            "humerus_l": 2.5,
            "scapula_r": 2.5,
            "scapula_l": 2.5,
            "calcn_r": 2.5,
            "calcn_l": 2.5,
            "talus_r": 2.5,
            "talus_l": 2.5
        }
        
        for joint, weight in problematic_joints.items():
            if joint in self.joint_indices:
                self.joint_weights[self.joint_indices[joint]] = weight
        
        # Define pose limits for better optimization
        self.pose_limits = kin_skel.pose_limits

        # SKEL pose parameter names for reference
        self.pose_param_names = kin_skel.pose_param_names
    
    def _init_parameters(self, target_joints, initial_poses=None, initial_trans=None):
        """
        Initialize parameters for optimization.
        
        Args:
            target_joints: Target anatomical joint positions [num_frames, 24, 3]
            initial_poses: Optional initial pose parameters [num_frames, num_pose_params]
            initial_trans: Optional initial translation parameters [num_frames, 3]
            
        Returns:
            Dictionary with initialized parameters
        """
        num_frames = target_joints.shape[0]
        
        # Initialize pose parameters
        if initial_poses is None:
            # Initialize with zeros for most parameters
            poses = torch.zeros((num_frames, self.num_pose_params), device=self.device)
            
            # Set some reasonable initial values for certain joints
            # Slight knee bend to avoid straight legs (which can cause optimization issues)
            poses[:, self.pose_param_names.index('knee_angle_r')] = 0.1
            poses[:, self.pose_param_names.index('knee_angle_l')] = 0.1
        else:
            poses = torch.tensor(initial_poses, dtype=torch.float32, device=self.device)
            
        # Initialize translation parameters
        if initial_trans is None:
            # Use target pelvis position as initial translation
            trans = target_joints[:, 0, :].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            trans = initial_trans.clone().detach().to(dtype=torch.float32, device=self.device)
            
        # Beta is fixed at zero for this fitter (as per requirements)
        betas = torch.zeros((num_frames, 10), device=self.device)
        
        return {
            'poses': poses,
            'betas': betas,
            'trans': trans,
            'target_joints': target_joints.clone().detach().to(dtype=torch.float32, device=self.device)
        }
    
    def fit_sequence(self, 
                     target_joints, 
                     initial_poses=None, 
                     initial_trans=None, 
                     max_iterations=20,
                     learning_rate=0.01,
                     pose_regularization=0.1):
        """
        Fit the SKEL model to a sequence of target joint positions.
        
        Args:
            target_joints: Target anatomical joint positions [num_frames, 24, 3]
            initial_poses: Optional initial pose parameters [num_frames, num_pose_params]
            initial_trans: Optional initial translation parameters [num_frames, 3]
            max_iterations: Maximum iterations for optimization
            learning_rate: Learning rate for the optimizer
            pose_regularization: Weight for pose regularization
            
        Returns:
            Dictionary with optimized parameters and metrics
        """
        num_frames = target_joints.shape[0]
        
        if self.debug: print(f"Fitting SKEL poses to {num_frames} frames of anatomical joint data")
        
        # Initialize parameters
        params = self._init_parameters(target_joints, initial_poses, initial_trans)
        
        # Initialize the output dictionary
        res_dict = {
            'poses': [],
            'trans': [],
            'betas': [],
            'joint_errors': [],
            'gender': self.gender
        }
        
        # Process all frames at once
        batch_params = {
            'poses': params['poses'].clone().requires_grad_(True),
            'betas': params['betas'].clone(),  # Fixed at zero, no gradient needed
            'trans': params['trans'].clone().requires_grad_(True),
            'target_joints': params['target_joints']
        }
        
        # Optimize the sequence
        batch_result = self._optimize_batch(batch_params, max_iterations, learning_rate, pose_regularization)
        
        # Store the results
        res_dict['poses'] = batch_result['poses'].detach().cpu().numpy()
        res_dict['trans'] = batch_result['trans'].detach().cpu().numpy()
        res_dict['betas'] = batch_result['betas'].detach().cpu().numpy()
        res_dict['joint_errors'] = batch_result['joint_errors'].detach().cpu().numpy()
        
        # Compute overall metrics
        mean_joint_error = np.mean(res_dict['joint_errors']) * 1000  # Convert to mm
        max_joint_error = np.max(res_dict['joint_errors']) * 1000  # Convert to mm
        
        if self.debug: print(f"Fitting complete. Mean joint error: {mean_joint_error:.2f} mm, Max joint error: {max_joint_error:.2f} mm")
        
        # If debug is enabled, plot the error distribution
        if self.debug:
            self._plot_error_distribution(res_dict['joint_errors'])
        
        return res_dict
    
    def _optimize_batch(self, batch_params, max_iterations, learning_rate, pose_regularization):
        """
        Optimize a batch of frames.
        
        Args:
            batch_params: Dictionary with batch parameters
            max_iterations: Maximum iterations for optimization
            learning_rate: Learning rate for the optimizer
            pose_regularization: Weight for pose regularization
            
        Returns:
            Dictionary with optimized parameters and metrics
        """
        # Create optimizer for pose and translation parameters
        optimizer = torch.optim.Adam([
            {'params': batch_params['poses']},
            {'params': batch_params['trans']}
        ], lr=learning_rate)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_iterations,
            eta_min=learning_rate * 0.01  # Minimum learning rate will be 1% of initial
        )
        
        # Keep track of the best result
        best_error = float('inf')
        best_params = {
            'poses': batch_params['poses'].clone().detach(),
            'trans': batch_params['trans'].clone().detach(),
            'betas': batch_params['betas'].clone().detach()
        }
        
        # Track losses and errors over iterations
        loss_history = []
        error_history = []
        
        # Early stopping variables
        min_error_improvement = 0.1  # Stop if error improvement is less than 0.1mm
        patience = 10  # Number of iterations to wait for improvement
        no_improvement_count = 0
        last_error_mm = float('inf')
        
        # Optimization loop
        for iter_idx in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass: compute SKEL joints
            output = self.skel.forward(
                poses=batch_params['poses'],
                betas=batch_params['betas'],
                trans=batch_params['trans'],
                poses_type='skel',
                skelmesh=False  # We don't need the mesh
            )
            
            # Compute losses
            loss_dict = self._compute_losses(
                output.joints, 
                batch_params['target_joints'],
                batch_params['poses'],
                pose_regularization
            )
            
            # Total loss
            total_loss = sum(loss_dict.values())
            
            # Track losses and errors
            loss_history.append(total_loss.item())
            error_history.append(loss_dict['joint_position_error'].item())
            
            # Backward pass
            total_loss.backward()
            
            # Apply pose limits
            with torch.no_grad():
                for param_idx, param_name in enumerate(self.pose_param_names):
                    if param_name in self.pose_limits:
                        min_val, max_val = self.pose_limits[param_name]
                        batch_params['poses'][:, param_idx].clamp_(min_val, max_val)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track best result
            current_error = loss_dict['joint_position_error'].item()
            if current_error < best_error:
                best_error = current_error
                best_params = {
                    'poses': batch_params['poses'].clone().detach(),
                    'trans': batch_params['trans'].clone().detach(),
                    'betas': batch_params['betas'].clone().detach()
                }
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Convert current error to millimeters for comparison
            current_error_mm = torch.sqrt(torch.tensor(current_error)) * 1000
            
            # Early stopping if error improvement is too small
            error_improvement = last_error_mm - current_error_mm
            if error_improvement < min_error_improvement and no_improvement_count >= patience:
                if self.debug:
                    print(f"  Early stopping at iteration {iter_idx} due to small error improvement ({error_improvement:.2f}mm)")
                break
                
            last_error_mm = current_error_mm
                
            # Debug output
            if self.debug:
                error_mm = torch.sqrt(torch.tensor(current_error)) * 1000  # Convert to mm
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Iteration {iter_idx}: Joint error: {error_mm:.2f} mm, LR: {current_lr:.6f}")
        
        # Plot loss and error history
        self._plot_optimization_history(loss_history, error_history)
        
        # Compute final joint errors per frame and per joint
        with torch.no_grad():
            output = self.skel.forward(
                poses=best_params['poses'],
                betas=best_params['betas'],
                trans=best_params['trans'],
                poses_type='skel',
                skelmesh=False
            )
            
            # Compute error per frame and per joint
            errors = torch.sqrt(((output.joints - batch_params['target_joints'])**2).sum(dim=2))  # [batch_size, 24]
            frame_errors = errors.mean(dim=1)  # Mean error across all joints, per frame
            
            # Compute and print per-joint errors
            joint_errors = errors.mean(dim=0)  # Mean error across all frames, per joint
            if self.debug:
                print("\nPer-joint errors (mm):")
                for joint_name, error in zip(self.joint_names, joint_errors.cpu().numpy() * 1000):
                    print(f"{joint_name:15s}: {error:.2f}")
        
        return {
            'poses': best_params['poses'],
            'trans': best_params['trans'],
            'betas': best_params['betas'],
            'joint_errors': frame_errors,
            'loss_history': loss_history,
            'error_history': error_history
        }
    
    def _compute_scapula_loss(self, poses):
        """
        Compute loss to regularize scapula movements.
        
        Args:
            poses: Pose parameters [batch_size, num_pose_params]
            
        Returns:
            Scapula regularization loss
        """
        # Scapula indices for right and left scapula
        scapula_indices = [26, 27, 28, 36, 37, 38]  # [right_abduction, right_elevation, right_rotation, left_abduction, left_elevation, left_rotation]
        
        scapula_poses = poses[:, scapula_indices]
        scapula_loss = torch.linalg.norm(scapula_poses, ord=2)
        return scapula_loss
    
    def _compute_spine_loss(self, poses):
        """
        Compute loss to regularize spine movements.
        
        Args:
            poses: Pose parameters [batch_size, num_pose_params]
            
        Returns:
            Spine regularization loss
        """
        # Spine indices (lumbar and thoracic)
        spine_indices = range(17, 25)  # Lumbar and thoracic spine parameters
        
        spine_poses = poses[:, spine_indices]
        spine_loss = torch.linalg.norm(spine_poses, ord=2)
        return spine_loss

    def _compute_losses(self, predicted_joints, target_joints, poses, pose_regularization):
        """
        Compute losses for optimization.
        
        Args:
            predicted_joints: Predicted joint positions [batch_size, 24, 3]
            target_joints: Target joint positions [batch_size, 24, 3]
            poses: Pose parameters [batch_size, num_pose_params]
            pose_regularization: Weight for pose regularization
            
        Returns:
            Dictionary of losses
        """
        # Joint position loss (weighted)
        weighted_sq_error = self.joint_weights * (predicted_joints - target_joints)**2
        joint_loss = weighted_sq_error.mean()
        
        # Pose regularization loss (to keep pose parameters small)
        # Exclude global rotation (first 3 parameters) from regularization
        pose_reg_loss = pose_regularization * (poses[:, 3:]**2).mean()
        
        # Scapula and spine regularization with adjusted weights
        scapula_loss = 0.2 * pose_regularization * self._compute_scapula_loss(poses)  # Increased from 0.1
        spine_loss = 0.15 * pose_regularization * self._compute_spine_loss(poses)  # Increased from 0.1
        
        # Encourage temporal consistency if we have multiple frames
        if poses.shape[0] > 1:
            time_loss = F.mse_loss(poses[1:], poses[:-1])
        else:
            time_loss = torch.tensor(0.0, device=self.device)
        
        # Compute per-joint position errors (for reporting)
        per_joint_error = ((predicted_joints - target_joints)**2).sum(dim=2)  # [batch_size, 24]
        joint_position_error = per_joint_error.mean()  # Mean across all joints and frames
        
        losses = {
            'joint_position_error': joint_position_error,
            'joint_loss': joint_loss,
            'pose_reg_loss': pose_reg_loss,
            'scapula_loss': scapula_loss,
            'spine_loss': spine_loss,
            'time_loss': 0.3 * time_loss
        }
        
        return losses
    
    def _plot_error_distribution(self, joint_errors):
        """
        Plot error distribution.
        
        Args:
            joint_errors: Joint errors [num_frames]
        """
        plt.figure(figsize=(10, 6))
        plt.hist(joint_errors * 1000, bins=50)  # Convert to mm
        plt.xlabel('Mean Joint Error (mm)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Joint Position Errors (Mean: {np.mean(joint_errors)*1000:.2f} mm)')
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'))
        plt.close()
        
    def visualize_results(self, target_joints, predicted_joints, frame_idx=0):
        """
        Visualize results for a specific frame.
        
        Args:
            target_joints: Target joint positions [num_frames, 24, 3]
            predicted_joints: Predicted joint positions [num_frames, 24, 3]
            frame_idx: Frame index to visualize
        """
        # This is a simple 2D visualization
        # For a full 3D visualization, you would need additional libraries
        
        # Extract the joints for the specified frame
        target = target_joints[frame_idx]
        pred = predicted_joints[frame_idx]
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot target joints
        ax1.scatter(target[:, 0], target[:, 1], c='blue', s=50)
        ax1.set_title('Target Joints')
        ax1.set_aspect('equal')
        
        # Plot predicted joints
        ax2.scatter(pred[:, 0], pred[:, 1], c='red', s=50)
        ax2.set_title('Predicted Joints')
        ax2.set_aspect('equal')
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, f'joints_comparison_frame_{frame_idx}.png'))
        plt.close()
        
    def save_results(self, results, filename="ik_results.pkl"):
        """
        Save results to a file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Create a copy with numpy arrays to ensure serializability
        save_dict = {
            'poses': results['poses'],
            'trans': results['trans'],
            'betas': results['betas'],
            'joint_errors': results['joint_errors'],
            'gender': results['gender']
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_dict, f)
            
        print(f"Results saved to {output_path}")
        
    def run_ik_from_file(self, 
               anatomical_joints_file, 
               output_file=None, 
               max_iterations=800,
               learning_rate=0.005,
               pose_regularization=0.03,
               trial=0,
               visualize_frame=0):
        """
        Run inverse kinematics on a file containing SMPL joint positions.
        
        Args:
            anatomical_joints_file: Path to file with SMPL joint positions
            output_file: Path to save results (default is ik_results.pkl in output_dir)
            max_iterations: Maximum iterations for optimization
            learning_rate: Learning rate for the optimizer
            pose_regularization: Weight for pose regularization
            trial: Trial index to process (for data with multiple trials)
            visualize_frame: Frame index to visualize (if debug is enabled)
            
        Returns:
            Dictionary with optimized parameters and metrics, including:
            - poses: Optimized pose parameters (PyTorch tensor)
            - trans: Optimized translation parameters (PyTorch tensor)
            - betas: Shape parameters (PyTorch tensor, fixed at zero)
            - joints: Final joint positions (PyTorch tensor)
            - joints_ori: Final joint orientations (PyTorch tensor)
            - joint_errors: Per-frame joint position errors (PyTorch tensor)
        """
        # Load the anatomical joint positions
        print(f"Loading anatomical joint data from {anatomical_joints_file}")
        anatomical_joints = np.load(anatomical_joints_file)
        
        # Check if we need to select a specific trial
        if isinstance(anatomical_joints, np.ndarray) and anatomical_joints.ndim == 4:
            # Shape is [trials, joints, dims, frames], need to transpose to [frames, joints, dims]
            anatomical_joints = anatomical_joints[trial].transpose(2, 0, 1)
            print(f"Loaded trial {trial}, shape after transpose: {anatomical_joints.shape}")
        else:
            print(f"Loaded anatomical joints, shape: {anatomical_joints.shape}")
        
        # Fit the sequence
        results = self.fit_sequence(
            anatomical_joints,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            pose_regularization=pose_regularization
        )
        
        # Get final joint positions and orientations
        with torch.no_grad():
            poses = torch.tensor(results['poses'], device=self.device)
            betas = torch.tensor(results['betas'], device=self.device)
            trans = torch.tensor(results['trans'], device=self.device)
            
            output = self.skel.forward(
                poses=poses,
                betas=betas,
                trans=trans,
                poses_type='skel',
                skelmesh=False
            )
            
            # Add joints and joints_ori to results as PyTorch tensors
            results['joints'] = output.joints  # Keep as tensor
            results['joints_ori'] = output.joints_ori  # Keep as tensor
            results['poses'] = poses  # Keep as tensor
            results['trans'] = trans  # Keep as tensor
            results['betas'] = betas  # Keep as tensor
            results['joint_errors'] = torch.tensor(results['joint_errors'], device=self.device)  # Convert to tensor
        
        # Visualize if debug is enabled
        if self.debug:
            predicted_joints = results['joints'].cpu().numpy()
            self.visualize_results(anatomical_joints, predicted_joints, frame_idx=visualize_frame)
        
        # Save results (convert tensors to numpy for saving)
        if output_file is None:
            output_file = "ik_results.pkl"
        save_dict = {
            'poses': results['poses'].cpu().numpy(),
            'trans': results['trans'].cpu().numpy(),
            'betas': results['betas'].cpu().numpy(),
            'joint_errors': results['joint_errors'].cpu().numpy(),
            'joints': results['joints'].cpu().numpy(),
            'joints_ori': results['joints_ori'].cpu().numpy(),
            'gender': self.gender
        }
        self.save_results(save_dict, filename=output_file)

        print("IK fitting complete!") 
        
        return results

    def run_ik(self, 
               anatomical_joints, 
               output_file=None, 
               max_iterations=800,
               learning_rate=0.005,
               pose_regularization=0.03,
               trial=0,
               visualize_frame=0):
        """
        Run inverse kinematics on a file containing SMPL joint positions.
        
        Args:
            anatomical_joints_file: Path to file with SMPL joint positions
            output_file: Path to save results (default is ik_results.pkl in output_dir)
            max_iterations: Maximum iterations for optimization
            learning_rate: Learning rate for the optimizer
            pose_regularization: Weight for pose regularization
            trial: Trial index to process (for data with multiple trials)
            visualize_frame: Frame index to visualize (if debug is enabled)
            
        Returns:
            Dictionary with optimized parameters and metrics, including:
            - poses: Optimized pose parameters (PyTorch tensor)
            - trans: Optimized translation parameters (PyTorch tensor)
            - betas: Shape parameters (PyTorch tensor, fixed at zero)
            - joints: Final joint positions (PyTorch tensor)
            - joints_ori: Final joint orientations (PyTorch tensor)
            - joint_errors: Per-frame joint position errors (PyTorch tensor)
        """

        # Check if we need to select a specific trial
        if isinstance(anatomical_joints, np.ndarray) and anatomical_joints.ndim == 4:
            # Shape is [trials, joints, dims, frames], need to transpose to [frames, joints, dims]
            anatomical_joints = anatomical_joints[trial].transpose(2, 0, 1)
            if self.debug: print(f"Loaded trial {trial}, shape after transpose: {anatomical_joints.shape}")
        else:
            if self.debug: print(f"Loaded anatomical joints, shape: {anatomical_joints.shape}")
        
        # Fit the sequence
        results = self.fit_sequence(
            anatomical_joints,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            pose_regularization=pose_regularization
        )
        
        # Get final joint positions and orientations
        with torch.no_grad():
            poses = torch.tensor(results['poses'], device=self.device)
            betas = torch.tensor(results['betas'], device=self.device)
            trans = torch.tensor(results['trans'], device=self.device)
            
            output = self.skel.forward(
                poses=poses,
                betas=betas,
                trans=trans,
                poses_type='skel',
                skelmesh=False
            )
            
            # Add joints and joints_ori to results as PyTorch tensors
            results['joints'] = output.joints  # Keep as tensor
            results['joints_ori'] = output.joints_ori  # Keep as tensor
            results['poses'] = poses  # Keep as tensor
            results['trans'] = trans  # Keep as tensor
            results['betas'] = betas  # Keep as tensor
            results['joint_errors'] = torch.tensor(results['joint_errors'], device=self.device)  # Convert to tensor
        
        # Visualize if debug is enabled
        if self.debug:
            predicted_joints = results['joints'].cpu().numpy()
            self.visualize_results(anatomical_joints, predicted_joints, frame_idx=visualize_frame)
        
        # Save results (convert tensors to numpy for saving)
        if output_file is not None:
            save_dict = {
                'poses': results['poses'].cpu().numpy(),
                'trans': results['trans'].cpu().numpy(),
                'betas': results['betas'].cpu().numpy(),
                'joint_errors': results['joint_errors'].cpu().numpy(),
                'joints': results['joints'].cpu().numpy(),
                'joints_ori': results['joints_ori'].cpu().numpy(),
                'gender': self.gender
            }
            self.save_results(save_dict, filename=output_file)

        if self.debug: print("IK fitting complete!") 
        
        return results

    def _plot_optimization_history(self, loss_history, error_history):
        """
        Plot the loss and error history during optimization.
        
        Args:
            loss_history: List of total loss values
            error_history: List of joint position error values
        """
        plt.figure(figsize=(12, 5))
        
        # Plot total loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title('Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot joint position error
        plt.subplot(1, 2, 2)
        plt.plot(error_history)
        plt.title('Joint Position Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimization_history.png'))
        plt.close()


if __name__ == "__main__":
    # Example usage
    fitter = AnatomicalJointFitter(
        gender="male",
        output_dir="Aurel/skeleton_fitting/output/ik_fitter",
        debug=True
    )
    
    # Run IK on anatomical joint file
    anatomical_joints_file = "Aurel/skeleton_fitting/output/regressor/regressed_joints.npy"
    
    results = fitter.run_ik(
        anatomical_joints_file=anatomical_joints_file,
        output_file="jumping_ik_results.pkl",
        max_iterations=150,
        learning_rate=0.1,
        pose_regularization=0.001,
        trial=0
    )
    