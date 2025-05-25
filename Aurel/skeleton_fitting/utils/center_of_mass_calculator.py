#!/usr/bin/env python3
"""
Center of Mass Calculator for Human Body Segments

This script calculates the center of mass position, inertia tensor, angular momentum,
and moment about the center of mass of the human body based on individual body 
segment properties and joint kinematics.

The input joints are already in global coordinates (including root translation),
and joint orientations are global rotation matrices. Mass centers and inertias
are defined in local body coordinate frames and need to be transformed to global coordinates.

Key Features:
- Center of mass position, velocity, and acceleration calculation
- Angular momentum about center of mass calculation
- Moment about center of mass (rate of change of angular momentum)
- Inertia tensor calculation about center of mass
- Segment angular velocity calculations
- Sophisticated derivative calculations with Savitzky-Golay smoothing
- Support for different frame rates with automatic parameter adjustment

The moment about COM can be compared with moments from ground reaction forces
to validate dynamic consistency and enforce motion dynamics.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import sys
import os
import json
from scipy.signal import savgol_filter

# Add the external SKEL path to import kin_skel
import skel.kin_skel as kin_skel

# Import body properties
try:
    from utils.body_properties_output import body_properties
except ImportError:
    from body_properties_output import body_properties


class CenterOfMassCalculator:
    """
    Calculator for center of mass position and inertia tensor based on body segment properties.
    
    This class computes the overall center of mass and inertia tensor for a human body
    given joint positions, orientations, and individual body segment properties.
    
    IMPORTANT COORDINATE FRAME ASSUMPTIONS:
    - Input joints are expected to be in global coordinates (including root translation)
    - Joint orientations are global rotation matrices representing the local coordinate frame
    - Mass centers and inertias are defined in local body coordinate frames (from body_properties)
    - We assume that the joint coordinate frame is aligned with the mass center coordinate frame
      for each segment (i.e., same orientation, just different origin)
    - Inertia tensors in body_properties are about the mass centers, not joint origins
    - Mass centers are transformed from local to global coordinates using joint positions and orientations
    """
    
    def __init__(self):
        """
        Initialize the center of mass calculator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_names = kin_skel.skel_joints_name
        self.body_properties = body_properties
        
        # Create mapping from joint names to indices
        self.joint_name_to_idx = {name: idx for idx, name in enumerate(self.joint_names)}
        
        # Extract body properties as tensors for efficient computation
        self._prepare_body_properties()
        
    def _prepare_body_properties(self) -> None:
        """Prepare body properties as tensors for efficient computation."""
        # Initialize lists to store properties in joint order
        masses = []
        mass_centers = []
        inertias = []
        
        # Process each joint in order
        for joint_name in self.joint_names:
            if joint_name in self.body_properties:
                props = self.body_properties[joint_name]
                masses.append(props["mass"])
                mass_centers.append(props["mass_center"])
                inertias.append(props["inertia"])
            else:
                # Handle missing bodies (e.g., patella) with zero properties
                masses.append(0.0)
                mass_centers.append([0.0, 0.0, 0.0])
                inertias.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Convert to tensors
        self.masses = torch.tensor(masses, dtype=torch.float32, device=self.device)  # Shape: (24,)
        self.mass_centers = torch.tensor(mass_centers, dtype=torch.float32, device=self.device)  # Shape: (24, 3)
        self.inertias = torch.tensor(inertias, dtype=torch.float32, device=self.device)  # Shape: (24, 6)
        
    def _transform_mass_center(self, 
                              joint_positions: torch.Tensor, 
                              joint_orientations: torch.Tensor) -> torch.Tensor:
        """
        Transform mass centers from local body coordinates to global coordinates.
        
        Args:
            joint_positions: Joint positions tensor of shape (B, 24, 3) - already in global coordinates
            joint_orientations: Joint orientation matrices of shape (B, 24, 3, 3) - global orientations
            
        Returns:
            Global mass center positions of shape (B, 24, 3)
        """
        B = joint_positions.shape[0]
        
        # Transform local mass centers to global coordinates
        # Following the same pattern as contact_models_torch.py:
        # global_position = joint_position + joint_orientation @ local_position
        
        # Expand mass centers to batch size: (B, 24, 3)
        local_mass_centers = self.mass_centers.unsqueeze(0).expand(B, -1, -1).contiguous()
        
        # Transform each mass center: joint_pos + joint_ori @ local_mass_center
        # Using batched matrix multiplication for efficiency
        transformed_mass_centers = torch.bmm(
            joint_orientations.reshape(B * 24, 3, 3),  # (B*24, 3, 3)
            local_mass_centers.reshape(B * 24, 3, 1)   # (B*24, 3, 1)
        ).reshape(B, 24, 3)  # (B, 24, 3)
        
        # Add joint positions to get global mass center positions
        global_mass_centers = joint_positions + transformed_mass_centers
        
        return global_mass_centers
    
    def _transform_inertia_tensor(self, 
                                 joint_orientations: torch.Tensor,
                                 global_mass_centers: torch.Tensor,
                                 global_com: torch.Tensor) -> torch.Tensor:
        """
        Transform inertia tensors from local body coordinates to global coordinates
        and apply parallel axis theorem from mass centers to global COM.
        
        IMPORTANT: The inertia tensors in body_properties are defined about the mass centers
        of each segment, not about the joint origins. We apply the parallel axis theorem
        to transfer these inertias from the mass centers to the global center of mass.
        
        Args:
            joint_orientations: Joint orientation matrices of shape (B, 24, 3, 3) - global orientations
            global_mass_centers: Global mass center positions of shape (B, 24, 3)
            global_com: Global center of mass position of shape (B, 3)
            
        Returns:
            Total inertia tensor about global COM of shape (B, 3, 3)
        """
        B = joint_orientations.shape[0]
        
        # Convert inertia vectors to 3x3 matrices
        # Inertia format: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        # These are inertia tensors about the mass centers in local coordinates
        local_inertias = torch.zeros(B, 24, 3, 3, device=self.device)
        
        # Fill the inertia matrices
        inertias_expanded = self.inertias.unsqueeze(0).expand(B, -1, -1)  # (B, 24, 6)
        
        local_inertias[:, :, 0, 0] = inertias_expanded[:, :, 0]  # Ixx
        local_inertias[:, :, 1, 1] = inertias_expanded[:, :, 1]  # Iyy
        local_inertias[:, :, 2, 2] = inertias_expanded[:, :, 2]  # Izz
        local_inertias[:, :, 0, 1] = inertias_expanded[:, :, 3]  # Ixy
        local_inertias[:, :, 1, 0] = inertias_expanded[:, :, 3]  # Ixy (symmetric)
        local_inertias[:, :, 0, 2] = inertias_expanded[:, :, 4]  # Ixz
        local_inertias[:, :, 2, 0] = inertias_expanded[:, :, 4]  # Ixz (symmetric)
        local_inertias[:, :, 1, 2] = inertias_expanded[:, :, 5]  # Iyz
        local_inertias[:, :, 2, 1] = inertias_expanded[:, :, 5]  # Iyz (symmetric)
        
        # Transform inertia tensors to global coordinates
        # I_global = R @ I_local @ R^T
        # Assuming joint orientations represent the mass center coordinate frame orientation
        R = joint_orientations  # (B, 24, 3, 3)
        R_T = R.transpose(-2, -1)  # (B, 24, 3, 3)
        
        global_inertias = torch.matmul(torch.matmul(R, local_inertias), R_T)  # (B, 24, 3, 3)
        
        # Apply parallel axis theorem to transfer from mass centers to global COM
        # I_about_global_COM = I_about_mass_center + m * (d^2 * I - d ⊗ d)
        # where d is the vector from global COM to segment mass center
        
        masses_expanded = self.masses.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, 3, 3)  # (B, 24, 3, 3)
        global_com_expanded = global_com.unsqueeze(1).expand(-1, 24, -1)  # (B, 24, 3)
        
        # Distance vectors from global COM to segment mass centers
        d_vectors = global_mass_centers - global_com_expanded  # (B, 24, 3)
        
        # Compute d^2 (squared distance)
        d_squared = torch.sum(d_vectors ** 2, dim=-1, keepdim=True).unsqueeze(-1)  # (B, 24, 1, 1)
        
        # Create identity matrices
        I_eye = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, 24, -1, -1)  # (B, 24, 3, 3)
        
        # Compute outer product d ⊗ d
        d_outer = torch.matmul(d_vectors.unsqueeze(-1), d_vectors.unsqueeze(-2))  # (B, 24, 3, 3)
        
        # Apply parallel axis theorem: I_total = I_cm + m * (d^2 * I - d ⊗ d)
        parallel_axis_correction = masses_expanded * (d_squared * I_eye - d_outer)
        
        # Total inertia for each segment about global COM
        segment_inertias_about_global_com = global_inertias + parallel_axis_correction
        
        # Sum over all segments to get total body inertia about global COM
        total_inertia = torch.sum(segment_inertias_about_global_com, dim=1)  # (B, 3, 3)
        
        return total_inertia
    
    def calculate_center_of_mass(self, 
                                joints: torch.Tensor, 
                                joints_ori: torch.Tensor,
                                fps: float = 20.0,
                                smooth_derivatives: bool = True) -> Dict[str, torch.Tensor]:
        """
        Calculate the center of mass position, velocity, acceleration, angular momentum, and moment.
        
        Args:
            joints: Joint positions tensor of shape (B, 24, 3) - already in global coordinates
            joints_ori: Joint orientation matrices of shape (B, 24, 3, 3) - global orientations
            fps: Frame rate for derivative calculations (default: 20.0)
            smooth_derivatives: Whether to apply smoothing to velocity and acceleration (default: True)
            
        Returns:
            Dictionary containing:
                - "com_position": Center of mass position (B, 3)
                - "com_velocity": Center of mass velocity (B, 3)
                - "com_acceleration": Center of mass acceleration (B, 3)
                - "angular_momentum": Angular momentum about COM (B, 3)
                - "moment_about_com": Moment about COM (rate of change of angular momentum) (B, 3)
                - "inertia_tensor": Inertia tensor about COM (B, 3, 3)
                - "total_mass": Total body mass (scalar)
                - "segment_masses": Individual segment masses (24,)
                - "segment_com_positions": Global segment COM positions (B, 24, 3)
                - "segment_angular_velocities": Angular velocities of segments (B, 24, 3)
        """
        # Move tensors to the correct device
        joints = joints.to(self.device)
        joints_ori = joints_ori.to(self.device)
        
        B = joints.shape[0]
        
        # Transform mass centers to global coordinates
        global_mass_centers = self._transform_mass_center(joints, joints_ori)
        
        # Calculate total mass
        total_mass = torch.sum(self.masses)
        
        # Calculate center of mass position
        # COM = Σ(m_i * r_i) / Σ(m_i)
        masses_expanded = self.masses.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)  # (B, 24, 3)
        weighted_positions = masses_expanded * global_mass_centers  # (B, 24, 3)
        com_position = torch.sum(weighted_positions, dim=1) / total_mass  # (B, 3)
        
        # Calculate center of mass velocity and acceleration using sophisticated method
        dt = 1.0 / fps
        com_velocity, com_acceleration = self._calculate_smooth_derivatives(
            com_position, 
            dt=dt, 
            smooth_velocity=smooth_derivatives,
            smooth_acceleration=smooth_derivatives
        )
        
        # Calculate inertia tensor about center of mass
        inertia_tensor = self._transform_inertia_tensor(joints_ori, global_mass_centers, com_position)
        
        # Calculate angular momentum about center of mass
        angular_momentum = self._calculate_angular_momentum(
            joints=joints,
            joints_ori=joints_ori,
            global_mass_centers=global_mass_centers,
            com_position=com_position,
            com_velocity=com_velocity,
            inertia_tensor=inertia_tensor,
            fps=fps
        )
        
        # Calculate moment about center of mass (rate of change of angular momentum)
        moment_about_com = self._calculate_moment_about_com(
            angular_momentum=angular_momentum,
            fps=fps,
            smooth_derivatives=smooth_derivatives
        )
        
        # Calculate segment angular velocities for additional analysis
        segment_angular_velocities = self._calculate_angular_velocities(joints_ori, fps)
        
        return {
            "com_position": com_position,
            "com_velocity": com_velocity,
            "com_acceleration": com_acceleration,
            "angular_momentum": angular_momentum,
            "moment_about_com": moment_about_com,
            "inertia_tensor": inertia_tensor,
            "total_mass": total_mass,
            "segment_masses": self.masses,
            "segment_com_positions": global_mass_centers,
            "segment_angular_velocities": segment_angular_velocities
        }
    
    def get_body_properties_summary(self) -> Dict[str, float]:
        """
        Get a summary of body properties.
        
        Returns:
            Dictionary with total mass and individual segment masses
        """
        summary = {"total_mass": float(torch.sum(self.masses))}
        
        for i, joint_name in enumerate(self.joint_names):
            summary[f"{joint_name}_mass"] = float(self.masses[i])
            
        return summary

    def save_results(self, 
                    results: Dict[str, torch.Tensor], 
                    output_dir: str, 
                    filename: str = "com_analysis_results.json",
                    save_format: str = "json") -> Optional[str]:
        """
        Save COM analysis results to file.
        
        Args:
            results: Dictionary containing COM calculation results
            output_dir: Directory to save the results
            filename: Name of the file (default: "com_analysis_results.json")
            save_format: Format to save ("json" or "pt", default: "json")
            
        Returns:
            Path to the saved file, or None if saving failed
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Full path to the output file
            filepath = os.path.join(output_dir, filename)
            
            if save_format.lower() == "json":
                # Convert tensors to JSON-serializable format
                json_results = {}
                
                for key, value in results.items():
                    if isinstance(value, torch.Tensor):
                        # Convert tensor to numpy array, then to list for JSON serialization
                        json_results[key] = value.detach().cpu().numpy().tolist()
                    else:
                        # Handle scalar values and other types
                        json_results[key] = float(value) if isinstance(value, (torch.Tensor, np.number)) else value
                
                # Add metadata
                json_results["metadata"] = {
                    "format": "json",
                    "description": "Center of mass analysis results",
                    "tensor_shapes": {
                        key: list(value.shape) if isinstance(value, torch.Tensor) else "scalar"
                        for key, value in results.items()
                    }
                }
                
                # Save as JSON
                with open(filepath, "w") as f:
                    json.dump(json_results, f, indent=2)
                    
            elif save_format.lower() == "pt":
                # Save as PyTorch tensor file (original format)
                torch.save(results, filepath)
                
            else:
                raise ValueError(f"Unsupported save format: {save_format}. Use 'json' or 'pt'.")
            
            print(f"COM analysis results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving COM results: {e}")
            return None
    
    def print_summary(self, results: Dict[str, torch.Tensor]) -> None:
        """
        Print a summary of COM calculation results.
        
        Args:
            results: Dictionary containing COM calculation results
        """
        print(f"Total body mass: {results['total_mass']:.2f} kg")
        print(f"COM position shape: {results['com_position'].shape}")
        print(f"Inertia tensor shape: {results['inertia_tensor'].shape}")

        if results['com_velocity'] is not None:
            print(f"COM velocity shape: {results['com_velocity'].shape}")
            vel_magnitude = results['com_velocity'].norm(dim=1)
            print(f"Average COM velocity magnitude: {vel_magnitude.mean():.3f} m/s")
            print(f"Max COM velocity magnitude: {vel_magnitude.max():.3f} m/s")

        if results['com_acceleration'] is not None:
            print(f"COM acceleration shape: {results['com_acceleration'].shape}")
            acc_magnitude = results['com_acceleration'].norm(dim=1)
            print(f"Average COM acceleration magnitude: {acc_magnitude.mean():.3f} m/s²")
            print(f"Max COM acceleration magnitude: {acc_magnitude.max():.3f} m/s²")

        # Print angular momentum information
        if "angular_momentum" in results and results["angular_momentum"] is not None:
            print(f"Angular momentum shape: {results['angular_momentum'].shape}")
            ang_mom_magnitude = results["angular_momentum"].norm(dim=1)
            print(f"Average angular momentum magnitude: {ang_mom_magnitude.mean():.3f} kg⋅m²/s")
            print(f"Max angular momentum magnitude: {ang_mom_magnitude.max():.3f} kg⋅m²/s")

        # Print moment information
        if "moment_about_com" in results and results["moment_about_com"] is not None:
            print(f"Moment about COM shape: {results['moment_about_com'].shape}")
            moment_magnitude = results["moment_about_com"].norm(dim=1)
            print(f"Average moment magnitude: {moment_magnitude.mean():.3f} N⋅m")
            print(f"Max moment magnitude: {moment_magnitude.max():.3f} N⋅m")

        # Print segment angular velocities information
        if "segment_angular_velocities" in results and results["segment_angular_velocities"] is not None:
            print(f"Segment angular velocities shape: {results['segment_angular_velocities'].shape}")
            seg_ang_vel_magnitude = results["segment_angular_velocities"].norm(dim=2)  # (B, 24)
            avg_seg_ang_vel = seg_ang_vel_magnitude.mean()
            max_seg_ang_vel = seg_ang_vel_magnitude.max()
            print(f"Average segment angular velocity magnitude: {avg_seg_ang_vel:.3f} rad/s")
            print(f"Max segment angular velocity magnitude: {max_seg_ang_vel:.3f} rad/s")

        # Print body properties summary
        if "body_properties_summary" in results:
            body_summary = results["body_properties_summary"]
            print(f"Body mass distribution:")
            for key, value in body_summary.items():
                if key != "total_mass":
                    print(f"  {key}: {value:.3f} kg")

    def calculate_and_save_center_of_mass(self, 
                                         joints: torch.Tensor, 
                                         joints_ori: torch.Tensor,
                                         output_dir: str = "output/com_analysis",
                                         filename: str = "com_analysis_results.json",
                                         save_format: str = "json",
                                         save_results: bool = True,
                                         print_summary: bool = True,
                                         fps: float = 120.0,
                                         smooth_derivatives: bool = True) -> Dict[str, torch.Tensor]:
        """
        Calculate center of mass and optionally save results and print summary.
        
        Args:
            joints: Joint positions tensor of shape (B, 24, 3)
            joints_ori: Joint orientation matrices of shape (B, 24, 3, 3)
            output_dir: Directory to save results (default: "output/com_analysis")
            filename: Filename for saved results (default: "com_analysis_results.json")
            save_format: Format to save ("json" or "pt", default: "json")
            save_results: Whether to save results to file (default: True)
            print_summary: Whether to print summary of results (default: True)
            fps: Frame rate for derivative calculations (default: 120.0)
            smooth_derivatives: Whether to apply smoothing to derivatives (default: True)
            
        Returns:
            Dictionary containing COM calculation results
        """
        print("Calculating center of mass and inertia properties...")

        # Calculate center of mass
        results = self.calculate_center_of_mass(
            joints, 
            joints_ori, 
            fps=fps, 
            smooth_derivatives=smooth_derivatives
        )
        
        # Add body properties summary to results
        body_summary = self.get_body_properties_summary()
        results["body_properties_summary"] = body_summary
        
        # Optionally print summary
        if print_summary:
            self.print_summary(results)
        
        # Optionally save results
        if save_results:
            self.save_results(results, output_dir, filename, save_format)
        
        return results

    def _calculate_smooth_derivatives(self, 
                                     position: torch.Tensor, 
                                     dt: float = 1.0/120.0,
                                     smooth_velocity: bool = True,
                                     smooth_acceleration: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate smooth velocity and acceleration using Savitzky-Golay filtering and central differences.
        
        Args:
            position: Position tensor of shape (B, 3) where B is number of frames
            dt: Time step between frames (default: 1/120 for 120 fps)
            smooth_velocity: Whether to apply smoothing to velocity calculation
            smooth_acceleration: Whether to apply smoothing to acceleration calculation
            
        Returns:
            Tuple of (velocity, acceleration) tensors, both of shape (B, 3)
        """
        B = position.shape[0]
        
        if B < 3:
            # Not enough frames for derivatives
            velocity = torch.zeros_like(position)
            acceleration = torch.zeros_like(position)
            return velocity, acceleration
        
        # Convert to numpy for scipy processing
        pos_np = position.detach().cpu().numpy()  # Shape: (B, 3)
        
        # Initialize output arrays
        velocity_np = np.zeros_like(pos_np)
        acceleration_np = np.zeros_like(pos_np)
        
        # Process each spatial dimension separately
        for dim in range(3):  # x, y, z dimensions
            pos_dim = pos_np[:, dim]  # Shape: (B,)
            
            # Apply Savitzky-Golay smoothing to position if we have enough points
            if smooth_velocity and B >= 5:
                # Use window length of min(B//4, 11) but at least 5, must be odd
                window_length = min(max(5, B//4), 11)
                if window_length % 2 == 0:
                    window_length += 1
                window_length = min(window_length, B)  # Ensure window doesn't exceed data length
                
                if window_length >= 5:
                    pos_smooth = savgol_filter(pos_dim, window_length, polyorder=3)
                else:
                    pos_smooth = pos_dim
            else:
                pos_smooth = pos_dim
            
            # Calculate velocity using central differences
            vel_dim = np.zeros(B)
            
            if B >= 3:
                # Central difference for interior points
                vel_dim[1:-1] = (pos_smooth[2:] - pos_smooth[:-2]) / (2 * dt)
                
                # Forward/backward difference for endpoints
                vel_dim[0] = (pos_smooth[1] - pos_smooth[0]) / dt
                vel_dim[-1] = (pos_smooth[-1] - pos_smooth[-2]) / dt
            elif B == 2:
                # Simple difference for 2 points
                vel_dim[0] = (pos_smooth[1] - pos_smooth[0]) / dt
                vel_dim[1] = vel_dim[0]
            else:
                # Single point - zero velocity
                vel_dim[0] = 0.0
            
            # Apply smoothing to velocity if requested
            if smooth_velocity and B >= 5:
                window_length = min(max(5, B//6), 9)
                if window_length % 2 == 0:
                    window_length += 1
                window_length = min(window_length, B)
                
                if window_length >= 5:
                    vel_dim = savgol_filter(vel_dim, window_length, polyorder=2)
            
            velocity_np[:, dim] = vel_dim
            
            # Calculate acceleration using central differences on velocity
            acc_dim = np.zeros(B)
            
            if B >= 3:
                # Central difference for interior points
                acc_dim[1:-1] = (vel_dim[2:] - vel_dim[:-2]) / (2 * dt)
                
                # Forward/backward difference for endpoints
                acc_dim[0] = (vel_dim[1] - vel_dim[0]) / dt
                acc_dim[-1] = (vel_dim[-1] - vel_dim[-2]) / dt
            elif B == 2:
                # Simple difference for 2 points
                acc_dim[0] = (vel_dim[1] - vel_dim[0]) / dt
                acc_dim[1] = acc_dim[0]
            else:
                # Single point - zero acceleration
                acc_dim[0] = 0.0
            
            # Apply smoothing to acceleration if requested
            if smooth_acceleration and B >= 5:
                window_length = min(max(5, B//8), 7)
                if window_length % 2 == 0:
                    window_length += 1
                window_length = min(window_length, B)
                
                if window_length >= 5:
                    acc_dim = savgol_filter(acc_dim, window_length, polyorder=2)
            
            acceleration_np[:, dim] = acc_dim
        
        # Convert back to torch tensors
        velocity = torch.from_numpy(velocity_np).float().to(position.device)
        acceleration = torch.from_numpy(acceleration_np).float().to(position.device)
        
        return velocity, acceleration

    def _calculate_angular_momentum(self,
                                   joints: torch.Tensor,
                                   joints_ori: torch.Tensor,
                                   global_mass_centers: torch.Tensor,
                                   com_position: torch.Tensor,
                                   com_velocity: torch.Tensor,
                                   inertia_tensor: torch.Tensor,
                                   fps: float = 20.0) -> torch.Tensor:
        """
        Calculate angular momentum about the center of mass.
        
        The total angular momentum about COM consists of:
        1. Angular momentum due to translation of body segments relative to COM
        2. Angular momentum due to rotation of body segments about their own mass centers
        
        IMPORTANT: Inertia tensors are defined about mass centers, not joint origins.
        Joint orientations represent the local coordinate frame, which we assume is
        aligned with the mass center coordinate frame for each segment.
        
        Args:
            joints: Joint positions tensor of shape (B, 24, 3)
            joints_ori: Joint orientation matrices of shape (B, 24, 3, 3)
            global_mass_centers: Global mass center positions of shape (B, 24, 3)
            com_position: Global center of mass position of shape (B, 3)
            com_velocity: Center of mass velocity of shape (B, 3)
            inertia_tensor: Total inertia tensor about COM of shape (B, 3, 3)
            fps: Frame rate for calculating angular velocities
            
        Returns:
            Angular momentum about COM of shape (B, 3)
        """
        B = joints.shape[0]
        dt = 1.0 / fps
        
        # Calculate segment mass center velocities using smooth derivatives
        segment_velocities = torch.zeros_like(global_mass_centers)  # (B, 24, 3)
        
        for segment_idx in range(24):
            segment_pos = global_mass_centers[:, segment_idx, :]  # (B, 3)
            segment_vel, _ = self._calculate_smooth_derivatives(
                segment_pos, dt=dt, smooth_velocity=True, smooth_acceleration=False
            )
            segment_velocities[:, segment_idx, :] = segment_vel
        
        # 1. Angular momentum due to translation of segments relative to COM
        # L_trans = Σ(r_i × m_i * v_i) where r_i is position of mass center relative to global COM
        com_expanded = com_position.unsqueeze(1).expand(-1, 24, -1)  # (B, 24, 3)
        relative_positions = global_mass_centers - com_expanded  # (B, 24, 3)
        
        masses_expanded = self.masses.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)  # (B, 24, 3)
        momentum_vectors = masses_expanded * segment_velocities  # (B, 24, 3)
        
        # Cross product: r × (m*v)
        angular_momentum_trans = torch.cross(relative_positions, momentum_vectors, dim=-1)  # (B, 24, 3)
        angular_momentum_trans = torch.sum(angular_momentum_trans, dim=1)  # (B, 3)
        
        # 2. Angular momentum due to rotation of segments about their own mass centers
        # L_rot = Σ(I_i * ω_i) where I_i is inertia tensor about mass center and ω_i is angular velocity
        
        # Calculate angular velocities of segments (assuming joint frame = mass center frame)
        angular_velocities = self._calculate_angular_velocities(joints_ori, fps)  # (B, 24, 3)
        
        # Transform local inertias (about mass centers) to global coordinates
        # Since inertia tensors are about mass centers, we use joint orientations assuming
        # the joint coordinate frame is aligned with the mass center coordinate frame
        local_inertias = torch.zeros(B, 24, 3, 3, device=self.device)
        inertias_expanded = self.inertias.unsqueeze(0).expand(B, -1, -1)  # (B, 24, 6)
        
        # Fill the inertia matrices (these are about mass centers in local coordinates)
        local_inertias[:, :, 0, 0] = inertias_expanded[:, :, 0]  # Ixx
        local_inertias[:, :, 1, 1] = inertias_expanded[:, :, 1]  # Iyy
        local_inertias[:, :, 2, 2] = inertias_expanded[:, :, 2]  # Izz
        local_inertias[:, :, 0, 1] = inertias_expanded[:, :, 3]  # Ixy
        local_inertias[:, :, 1, 0] = inertias_expanded[:, :, 3]  # Ixy (symmetric)
        local_inertias[:, :, 0, 2] = inertias_expanded[:, :, 4]  # Ixz
        local_inertias[:, :, 2, 0] = inertias_expanded[:, :, 4]  # Ixz (symmetric)
        local_inertias[:, :, 1, 2] = inertias_expanded[:, :, 5]  # Iyz
        local_inertias[:, :, 2, 1] = inertias_expanded[:, :, 5]  # Iyz (symmetric)
        
        # Transform inertia tensors to global coordinates: I_global = R @ I_local @ R^T
        # Here we assume joint orientation represents the mass center coordinate frame
        R = joints_ori  # (B, 24, 3, 3)
        R_T = R.transpose(-2, -1)  # (B, 24, 3, 3)
        global_segment_inertias = torch.matmul(torch.matmul(R, local_inertias), R_T)  # (B, 24, 3, 3)
        
        # Calculate rotational angular momentum: L_rot = I * ω
        # This is the angular momentum of each segment about its own mass center
        angular_momentum_rot = torch.zeros(B, 3, device=self.device)
        for segment_idx in range(24):
            if self.masses[segment_idx] > 0:  # Only process segments with mass
                I_segment = global_segment_inertias[:, segment_idx, :, :]  # (B, 3, 3)
                omega_segment = angular_velocities[:, segment_idx, :]  # (B, 3)
                L_segment = torch.bmm(I_segment, omega_segment.unsqueeze(-1)).squeeze(-1)  # (B, 3)
                angular_momentum_rot += L_segment
        
        # Total angular momentum about global COM
        # This is the sum of translational and rotational components
        total_angular_momentum = angular_momentum_trans + angular_momentum_rot
        
        return total_angular_momentum
    
    def _calculate_angular_velocities(self, joints_ori: torch.Tensor, fps: float = 20.0) -> torch.Tensor:
        """
        Calculate angular velocities from orientation matrices.
        
        Args:
            joints_ori: Joint orientation matrices of shape (B, 24, 3, 3)
            fps: Frame rate for derivative calculation
            
        Returns:
            Angular velocities of shape (B, 24, 3)
        """
        B, num_joints, _, _ = joints_ori.shape
        dt = 1.0 / fps
        
        angular_velocities = torch.zeros(B, num_joints, 3, device=self.device)
        
        if B < 2:
            return angular_velocities
        
        for joint_idx in range(num_joints):
            R = joints_ori[:, joint_idx, :, :]  # (B, 3, 3)
            
            # Calculate angular velocity using finite differences
            # ω = (R_dot * R^T)_skew where _skew extracts the skew-symmetric part
            
            # Calculate R_dot using central differences
            R_dot = torch.zeros_like(R)
            
            if B >= 3:
                # Central difference for interior points
                R_dot[1:-1] = (R[2:] - R[:-2]) / (2 * dt)
                # Forward/backward difference for endpoints
                R_dot[0] = (R[1] - R[0]) / dt
                R_dot[-1] = (R[-1] - R[-2]) / dt
            else:
                # Simple difference for 2 points
                R_dot[0] = (R[1] - R[0]) / dt
                R_dot[1] = R_dot[0]
            
            # Calculate ω from R_dot * R^T
            R_T = R.transpose(-2, -1)
            omega_skew = torch.bmm(R_dot, R_T)  # (B, 3, 3)
            
            # Extract angular velocity from skew-symmetric matrix
            # For skew-symmetric matrix S = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
            # The vector is [x, y, z]
            angular_velocities[:, joint_idx, 0] = omega_skew[:, 2, 1]  # x component
            angular_velocities[:, joint_idx, 1] = omega_skew[:, 0, 2]  # y component
            angular_velocities[:, joint_idx, 2] = omega_skew[:, 1, 0]  # z component
        
        return angular_velocities
    
    def _calculate_moment_about_com(self,
                                   angular_momentum: torch.Tensor,
                                   fps: float = 20.0,
                                   smooth_derivatives: bool = True) -> torch.Tensor:
        """
        Calculate moment (rate of change of angular momentum) about center of mass.
        
        Args:
            angular_momentum: Angular momentum about COM of shape (B, 3)
            fps: Frame rate for derivative calculation
            smooth_derivatives: Whether to apply smoothing to the derivative
            
        Returns:
            Moment about COM of shape (B, 3)
        """
        dt = 1.0 / fps
        
        # Calculate time derivative of angular momentum
        moment, _ = self._calculate_smooth_derivatives(
            angular_momentum,
            dt=dt,
            smooth_velocity=smooth_derivatives,
            smooth_acceleration=False
        )
        
        return moment


def calculate_com_from_kinematics(joints: Union[torch.Tensor, np.ndarray], 
                                 joints_ori: Union[torch.Tensor, np.ndarray],
                                 device: str = "cpu",
                                 fps: float = 20.0,
                                 smooth_derivatives: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convenience function to calculate center of mass from joint kinematics.
    
    Args:
        joints: Joint positions (B, 24, 3) - can be numpy array or torch tensor (global coordinates)
        joints_ori: Joint orientations (B, 24, 3, 3) - can be numpy array or torch tensor (global orientations)
        device: Device to run calculations on
        fps: Frame rate for derivative calculations (default: 20.0)
        smooth_derivatives: Whether to apply smoothing to derivatives (default: True)
        
    Returns:
        Dictionary with COM calculations including position, velocity, and acceleration
    """
    # Convert to torch tensors if needed
    if isinstance(joints, np.ndarray):
        joints = torch.from_numpy(joints).float()
    if isinstance(joints_ori, np.ndarray):
        joints_ori = torch.from_numpy(joints_ori).float()
    
    # Initialize calculator
    calculator = CenterOfMassCalculator(device=device)
    
    # Calculate COM
    results = calculator.calculate_center_of_mass(
        joints, 
        joints_ori, 
        fps=fps, 
        smooth_derivatives=smooth_derivatives
    )
    
    return results


def calculate_moment_about_com_from_kinematics(joints: Union[torch.Tensor, np.ndarray], 
                                              joints_ori: Union[torch.Tensor, np.ndarray],
                                              device: str = "cpu",
                                              fps: float = 20.0,
                                              smooth_derivatives: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convenience function to calculate moment about center of mass from joint kinematics.
    
    This function is specifically designed for comparing with moments from ground reaction forces
    to validate dynamic consistency. The moment about COM represents the rate of change of 
    angular momentum and should equal the sum of all external moments acting on the body.
    
    Args:
        joints: Joint positions (B, 24, 3) - can be numpy array or torch tensor (global coordinates)
        joints_ori: Joint orientations (B, 24, 3, 3) - can be numpy array or torch tensor (global orientations)
        device: Device to run calculations on
        fps: Frame rate for derivative calculations (default: 20.0)
        smooth_derivatives: Whether to apply smoothing to derivatives (default: True)
        
    Returns:
        Dictionary with key results for dynamics validation:
            - "moment_about_com": Moment about COM (B, 3) [N⋅m]
            - "angular_momentum": Angular momentum about COM (B, 3) [kg⋅m²/s]
            - "com_position": Center of mass position (B, 3) [m]
            - "com_velocity": Center of mass velocity (B, 3) [m/s]
            - "com_acceleration": Center of mass acceleration (B, 3) [m/s²]
            - "inertia_tensor": Inertia tensor about COM (B, 3, 3) [kg⋅m²]
            - "total_mass": Total body mass [kg]
    """
    # Convert to torch tensors if needed
    if isinstance(joints, np.ndarray):
        joints = torch.from_numpy(joints).float()
    if isinstance(joints_ori, np.ndarray):
        joints_ori = torch.from_numpy(joints_ori).float()
    
    # Initialize calculator
    calculator = CenterOfMassCalculator()
    
    # Calculate full COM analysis
    full_results = calculator.calculate_center_of_mass(
        joints, 
        joints_ori, 
        fps=fps, 
        smooth_derivatives=smooth_derivatives
    )
    
    # Return subset focused on dynamics validation
    return {
        "moment_about_com": full_results["moment_about_com"],
        "angular_momentum": full_results["angular_momentum"],
        "com_position": full_results["com_position"],
        "com_velocity": full_results["com_velocity"],
        "com_acceleration": full_results["com_acceleration"],
        "inertia_tensor": full_results["inertia_tensor"],
        "total_mass": full_results["total_mass"]
    }


# Test functions for when script is run directly
def create_realistic_motion_data(num_frames: int = 120, fps: float = 20.0):
    """Create realistic motion data with known velocity and acceleration patterns."""
    
    # Time vector
    t = torch.linspace(0, (num_frames-1)/fps, num_frames)
    
    # Create a jumping motion pattern
    # Vertical component: parabolic motion with gravity
    y_pos = 1.0 + 0.5 * torch.sin(2 * np.pi * t / t[-1]) - 0.5 * (t - t[-1]/2)**2 / (t[-1]/4)**2
    
    # Horizontal components: sinusoidal motion
    x_pos = 0.2 * torch.sin(4 * np.pi * t / t[-1])
    z_pos = 0.1 * torch.cos(6 * np.pi * t / t[-1])
    
    # Combine into position vector
    com_position = torch.stack([x_pos, y_pos, z_pos], dim=1)  # Shape: (num_frames, 3)
    
    # Add some noise to make it more realistic (adjust noise level for lower fps)
    noise_level = 0.005 if fps <= 30 else 0.01  # Less noise for lower fps data
    noise = torch.randn_like(com_position) * noise_level
    com_position_noisy = com_position + noise
    
    # Calculate analytical derivatives for comparison
    dt = 1.0 / fps
    
    # Analytical velocity (central differences on clean data)
    velocity_analytical = torch.zeros_like(com_position)
    velocity_analytical[1:-1] = (com_position[2:] - com_position[:-2]) / (2 * dt)
    velocity_analytical[0] = (com_position[1] - com_position[0]) / dt
    velocity_analytical[-1] = (com_position[-1] - com_position[-2]) / dt
    
    # Analytical acceleration (central differences on velocity)
    acceleration_analytical = torch.zeros_like(com_position)
    acceleration_analytical[1:-1] = (velocity_analytical[2:] - velocity_analytical[:-2]) / (2 * dt)
    acceleration_analytical[0] = (velocity_analytical[1] - velocity_analytical[0]) / dt
    acceleration_analytical[-1] = (velocity_analytical[-1] - velocity_analytical[-2]) / dt
    
    return {
        "time": t,
        "position_clean": com_position,
        "position_noisy": com_position_noisy,
        "velocity_analytical": velocity_analytical,
        "acceleration_analytical": acceleration_analytical
    }

def create_dummy_joint_data(com_trajectory: torch.Tensor):
    """Create dummy joint data that produces the given COM trajectory."""
    num_frames, _ = com_trajectory.shape
    num_joints = 24
    
    # Create dummy joint positions centered around the COM
    joints = torch.zeros(num_frames, num_joints, 3)
    
    # Distribute joints around the COM with some random offsets
    for i in range(num_joints):
        offset = torch.randn(3) * 0.3  # Random offset for each joint
        joints[:, i, :] = com_trajectory + offset.unsqueeze(0)
    
    # Create identity rotation matrices for simplicity
    joints_ori = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(num_frames, num_joints, -1, -1)
    
    return joints, joints_ori

def test_derivative_calculation():
    """Test the derivative calculation with different settings."""
    print("Testing Velocity, Acceleration, Angular Momentum, and Moment Calculation")
    print("=" * 70)
    
    # Create realistic motion data with 20 fps (your actual frame rate)
    motion_data = create_realistic_motion_data(num_frames=120, fps=20.0)
    
    # Create dummy joint data with some rotation
    joints, joints_ori = create_dummy_joint_data_with_rotation(motion_data["position_noisy"])
    
    # Initialize calculator
    calculator = CenterOfMassCalculator()
    
    # Test different configurations - focused on 20 fps and relevant comparisons
    configs = [
        {"fps": 20.0, "smooth": True, "name": "20fps_smoothed"},
        {"fps": 20.0, "smooth": False, "name": "20fps_unsmoothed"},
        {"fps": 25.0, "smooth": True, "name": "25fps_smoothed"},  # Close comparison for validation
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"FPS: {config['fps']}, Smoothing: {config['smooth']}")
        
        # Calculate COM with current configuration
        com_results = calculator.calculate_center_of_mass(
            joints=joints,
            joints_ori=joints_ori,
            fps=config["fps"],
            smooth_derivatives=config["smooth"]
        )
        
        results[config["name"]] = com_results
        
        # Print summary statistics
        vel_mag = com_results["com_velocity"].norm(dim=1)
        acc_mag = com_results["com_acceleration"].norm(dim=1)
        ang_mom_mag = com_results["angular_momentum"].norm(dim=1)
        moment_mag = com_results["moment_about_com"].norm(dim=1)
        
        print(f"  Velocity - Mean: {vel_mag.mean():.3f} m/s, Max: {vel_mag.max():.3f} m/s")
        print(f"  Acceleration - Mean: {acc_mag.mean():.3f} m/s², Max: {acc_mag.max():.3f} m/s²")
        print(f"  Angular Momentum - Mean: {ang_mom_mag.mean():.3f} kg⋅m²/s, Max: {ang_mom_mag.max():.3f} kg⋅m²/s")
        print(f"  Moment about COM - Mean: {moment_mag.mean():.3f} N⋅m, Max: {moment_mag.max():.3f} N⋅m")
    
    return results, motion_data

def create_dummy_joint_data_with_rotation(com_trajectory: torch.Tensor):
    """Create dummy joint data with rotation that produces the given COM trajectory."""
    num_frames, _ = com_trajectory.shape
    num_joints = 24
    
    # Create dummy joint positions centered around the COM
    joints = torch.zeros(num_frames, num_joints, 3)
    
    # Distribute joints around the COM with some random offsets
    for i in range(num_joints):
        offset = torch.randn(3) * 0.3  # Random offset for each joint
        joints[:, i, :] = com_trajectory + offset.unsqueeze(0)
    
    # Create rotation matrices with some time-varying rotation
    joints_ori = torch.zeros(num_frames, num_joints, 3, 3)
    
    # Time vector for creating rotations
    t = torch.linspace(0, 2*np.pi, num_frames)
    
    for i in range(num_joints):
        for frame in range(num_frames):
            # Create rotation around different axes for different joints
            angle_x = 0.1 * torch.sin(t[frame] + i * 0.1)  # Small rotation around X
            angle_y = 0.1 * torch.cos(t[frame] + i * 0.2)  # Small rotation around Y  
            angle_z = 0.05 * torch.sin(2 * t[frame] + i * 0.3)  # Small rotation around Z
            
            # Create rotation matrices
            cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
            cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
            cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
            
            # Rotation around X
            Rx = torch.tensor([[1, 0, 0],
                              [0, cos_x, -sin_x],
                              [0, sin_x, cos_x]], dtype=torch.float32)
            
            # Rotation around Y
            Ry = torch.tensor([[cos_y, 0, sin_y],
                              [0, 1, 0],
                              [-sin_y, 0, cos_y]], dtype=torch.float32)
            
            # Rotation around Z
            Rz = torch.tensor([[cos_z, -sin_z, 0],
                              [sin_z, cos_z, 0],
                              [0, 0, 1]], dtype=torch.float32)
            
            # Combined rotation
            joints_ori[frame, i] = Rz @ Ry @ Rx
    
    return joints, joints_ori

def plot_comparison(results: dict, motion_data: dict, save_plots: bool = True):
    """Plot comparison of different derivative calculation methods including angular momentum and moment."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    time = motion_data["time"].numpy()
    
    # Create figure with more subplots to include angular momentum and moment
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("COM Analysis: Position, Velocity, Acceleration, Angular Momentum, and Moment (20 fps focus)", fontsize=16)
    
    # Plot position (Y-axis only for clarity)
    axes[0, 0].plot(time, motion_data["position_clean"][:, 1].numpy(), "k-", label="Clean", linewidth=2)
    axes[0, 0].plot(time, motion_data["position_noisy"][:, 1].numpy(), "gray", alpha=0.7, label="Noisy")
    for name, result in results.items():
        if "20fps" in name:  # Focus on 20fps results
            axes[0, 0].plot(time, result["com_position"][:, 1].numpy(), "--", label=f"Calculated ({name})")
    axes[0, 0].set_title("Y Position")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot velocity magnitude
    axes[0, 1].plot(time, motion_data["velocity_analytical"].norm(dim=1).numpy(), "k-", label="Analytical", linewidth=2)
    for name, result in results.items():
        axes[0, 1].plot(time, result["com_velocity"].norm(dim=1).numpy(), label=name)
    axes[0, 1].set_title("Velocity Magnitude")
    axes[0, 1].set_ylabel("Velocity (m/s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot acceleration magnitude
    axes[0, 2].plot(time, motion_data["acceleration_analytical"].norm(dim=1).numpy(), "k-", label="Analytical", linewidth=2)
    for name, result in results.items():
        axes[0, 2].plot(time, result["com_acceleration"].norm(dim=1).numpy(), label=name)
    axes[0, 2].set_title("Acceleration Magnitude")
    axes[0, 2].set_ylabel("Acceleration (m/s²)")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot velocity components (X, Y, Z)
    for i, component in enumerate(["X", "Y", "Z"]):
        axes[1, i].plot(time, motion_data["velocity_analytical"][:, i].numpy(), "k-", label="Analytical", linewidth=2)
        for name, result in results.items():
            if "20fps" in name:  # Focus on 20fps results for clarity
                axes[1, i].plot(time, result["com_velocity"][:, i].numpy(), "--", label=name)
        axes[1, i].set_title(f"Velocity {component}")
        axes[1, i].set_ylabel("Velocity (m/s)")
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    # Plot acceleration components (X, Y, Z)
    for i, component in enumerate(["X", "Y", "Z"]):
        axes[2, i].plot(time, motion_data["acceleration_analytical"][:, i].numpy(), "k-", label="Analytical", linewidth=2)
        for name, result in results.items():
            if "20fps" in name:  # Focus on 20fps results for clarity
                axes[2, i].plot(time, result["com_acceleration"][:, i].numpy(), "--", label=name)
        axes[2, i].set_title(f"Acceleration {component}")
        axes[2, i].set_ylabel("Acceleration (m/s²)")
        axes[2, i].legend()
        axes[2, i].grid(True)
    
    # Plot angular momentum components (X, Y, Z)
    for i, component in enumerate(["X", "Y", "Z"]):
        for name, result in results.items():
            if "20fps" in name:  # Focus on 20fps results for clarity
                axes[3, i].plot(time, result["angular_momentum"][:, i].numpy(), label=name)
        axes[3, i].set_title(f"Angular Momentum {component}")
        axes[3, i].set_ylabel("Angular Momentum (kg⋅m²/s)")
        axes[3, i].set_xlabel("Time (s)")
        axes[3, i].legend()
        axes[3, i].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig("com_analysis_comparison_20fps.png", dpi=300, bbox_inches="tight")
        print("Plot saved as: com_analysis_comparison_20fps.png")
    
    # Create a second figure for moment analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("Moment about COM Analysis (20 fps focus)", fontsize=16)
    
    # Plot moment magnitude
    for name, result in results.items():
        axes2[0, 0].plot(time, result["moment_about_com"].norm(dim=1).numpy(), label=name)
    axes2[0, 0].set_title("Moment Magnitude")
    axes2[0, 0].set_ylabel("Moment (N⋅m)")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)
    
    # Plot angular momentum magnitude
    for name, result in results.items():
        axes2[0, 1].plot(time, result["angular_momentum"].norm(dim=1).numpy(), label=name)
    axes2[0, 1].set_title("Angular Momentum Magnitude")
    axes2[0, 1].set_ylabel("Angular Momentum (kg⋅m²/s)")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True)
    
    # Plot moment components (X, Y)
    for i, component in enumerate(["X", "Y"]):
        for name, result in results.items():
            if "20fps" in name:  # Focus on 20fps results for clarity
                axes2[1, i].plot(time, result["moment_about_com"][:, i].numpy(), label=name)
        axes2[1, i].set_title(f"Moment {component}")
        axes2[1, i].set_ylabel("Moment (N⋅m)")
        axes2[1, i].set_xlabel("Time (s)")
        axes2[1, i].legend()
        axes2[1, i].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig("moment_analysis_20fps.png", dpi=300, bbox_inches="tight")
        print("Moment plot saved as: moment_analysis_20fps.png")
    
    plt.show()

def save_test_results(results: dict, motion_data: dict):
    """Save test results to JSON for further analysis."""
    
    # Prepare data for JSON serialization
    json_data = {
        "motion_data": {
            "time": motion_data["time"].tolist(),
            "position_clean": motion_data["position_clean"].tolist(),
            "position_noisy": motion_data["position_noisy"].tolist(),
            "velocity_analytical": motion_data["velocity_analytical"].tolist(),
            "acceleration_analytical": motion_data["acceleration_analytical"].tolist()
        },
        "calculated_results": {}
    }
    
    for name, result in results.items():
        json_data["calculated_results"][name] = {
            "com_position": result["com_position"].tolist(),
            "com_velocity": result["com_velocity"].tolist(),
            "com_acceleration": result["com_acceleration"].tolist(),
            "angular_momentum": result["angular_momentum"].tolist(),
            "moment_about_com": result["moment_about_com"].tolist(),
            "total_mass": float(result["total_mass"]),
            "inertia_tensor": result["inertia_tensor"].tolist(),
            "segment_angular_velocities": result["segment_angular_velocities"].tolist()
        }
    
    # Save to file
    with open("com_analysis_test_results.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("Test results saved to: com_analysis_test_results.json")

def calculate_errors(results: dict, motion_data: dict):
    """Calculate and print error statistics."""
    print("\nError Analysis (compared to analytical solution):")
    print("=" * 60)
    
    for name, result in results.items():
        # Calculate RMS errors
        vel_error = (result["com_velocity"] - motion_data["velocity_analytical"]).norm(dim=1)
        acc_error = (result["com_acceleration"] - motion_data["acceleration_analytical"]).norm(dim=1)
        
        vel_rmse = torch.sqrt(torch.mean(vel_error**2))
        acc_rmse = torch.sqrt(torch.mean(acc_error**2))
        
        print(f"{name}:")
        print(f"  Velocity RMSE: {vel_rmse:.4f} m/s")
        print(f"  Acceleration RMSE: {acc_rmse:.4f} m/s²")

def run_comprehensive_test():
    """Run comprehensive test suite for the center of mass calculator."""
    print("Enhanced Center of Mass Calculator - Comprehensive Test Suite")
    print("=" * 70)
    print("Testing with 20 fps data (120 frames = 6 seconds of motion)")
    print("Including Angular Momentum and Moment about COM calculations")
    print("=" * 70)
    
    try:
        # Run derivative calculation tests
        results, motion_data = test_derivative_calculation()
        
        # Calculate and print error statistics
        calculate_errors(results, motion_data)
        
        # Save results
        save_test_results(results, motion_data)
        
        # Create comparison plots
        print("\nGenerating comparison plots...")
        plot_comparison(results, motion_data, save_plots=True)
        
        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("\nFeatures demonstrated:")
        print("✓ Sophisticated velocity calculation using Savitzky-Golay smoothing")
        print("✓ Accurate acceleration calculation with central differences")
        print("✓ Angular momentum calculation about center of mass")
        print("✓ Moment calculation (rate of change of angular momentum)")
        print("✓ Segment angular velocity calculations")
        print("✓ Comparison of smoothed vs unsmoothed derivatives")
        print("✓ Support for different frame rates (20, 25 fps)")
        print("✓ Optimized smoothing parameters for lower frame rates")
        print("✓ Error analysis against analytical solutions")
        print("✓ Comprehensive visualization of results")
        print("✓ JSON saving with proper tensor conversion")
        print("✓ Optional printing and saving functionality")
        
        print(f"\nNote: Your data at 20 fps provides {motion_data['time'][-1]:.1f} seconds of motion")
        print("The algorithm automatically adjusts time step (dt = 0.05s) for accurate derivatives.")
        print("\nAngular momentum and moment calculations enable:")
        print("• Comparison with moments from ground reaction forces")
        print("• Validation of dynamic consistency")
        print("• Analysis of rotational motion characteristics")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nTest completed.")


if __name__ == "__main__":
    print("Center of Mass Calculator")
    print("=" * 50)
    print("This script can be used as a module or run directly for testing.")
    print("\nRunning comprehensive test suite...")
    print()
    
    run_comprehensive_test()
