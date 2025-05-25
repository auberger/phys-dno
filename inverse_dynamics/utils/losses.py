#!/usr/bin/env python3
"""
Dynamical Consistency Losses for Biomechanical Motion Analysis

This module implements loss functions to enforce Newton's laws of motion for human body dynamics.
The losses compare computed center of mass dynamics with ground reaction forces to ensure
physical consistency of the motion.

Key Features:
- Translational consistency: F = ma (Newton's second law for linear motion)
- Rotational consistency: τ = dL/dt (Newton's second law for angular motion)
- GPU-compatible PyTorch implementation
- Flexible weighting and normalization options

Usage:
    from Aurel.inverse_dynamics.losses import calculate_dynamical_consistency_losses
    
    loss = calculate_dynamical_consistency_losses(
        joints=joints,
        joints_ori=joints_ori,
        contact_output=contact_output,
        fps=20.0
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union

# Import required modules
from inverse_dynamics.utils.center_of_mass_calculator import CenterOfMassCalculator
from inverse_dynamics.utils.contact_models_torch import ContactOutput

class DynamicalConsistencyLoss(nn.Module):
    """
    Loss function for enforcing dynamical consistency in biomechanical motion.
    
    Implements two main loss terms:
    1. Translational consistency: L_trans = ||m*(a - g) - F_GRF||²
    2. Rotational consistency: L_rot = ||τ_GRF_about_COM - dL/dt||²
    """
    
    def __init__(self,
                 gravity: float = 9.81,
                 translational_weight: float = 1.0,
                 rotational_weight: float = 1.0,
                 normalize_by_mass: bool = True,
                 normalize_by_time: bool = True,
                 reduction: str = "mean"):
        """
        Initialize the dynamical consistency loss.
        
        Args:
            gravity: Gravitational acceleration magnitude (m/s²) (default: 9.81)
            translational_weight: Weight for translational consistency loss (default: 1.0)
            rotational_weight: Weight for rotational consistency loss (default: 1.0)
            normalize_by_mass: Whether to normalize forces by body mass (default: True)
            normalize_by_time: Whether to normalize by time step for rate terms (default: True)
            reduction: Reduction method for loss ("mean", "sum", "none") (default: "mean")
        """
        super().__init__()
        
        self.gravity = gravity
        self.translational_weight = translational_weight
        self.rotational_weight = rotational_weight
        self.normalize_by_mass = normalize_by_mass
        self.normalize_by_time = normalize_by_time
        self.reduction = reduction
        
        # Gravity vector (pointing downward in Y direction)
        self.register_buffer("gravity_vector", torch.tensor([0.0, -gravity, 0.0]))
        
        # Initialize COM calculator for internal use
        self.com_calculator = CenterOfMassCalculator()
        
        # Loss function
        if reduction == "mean":
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif reduction == "sum":
            self.loss_fn = nn.MSELoss(reduction="sum")
        else:
            self.loss_fn = nn.MSELoss(reduction="none")
    
    def forward(self,
                joints: torch.Tensor,
                joints_ori: torch.Tensor,
                contact_output: ContactOutput,
                fps: float = 20.0,
                com_results: Optional[Dict[str, torch.Tensor]] = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate dynamical consistency losses.
        
        Args:
            joints: Joint positions (B, 24, 3) in global coordinates
            joints_ori: Joint orientations (B, 24, 3, 3) as rotation matrices
            contact_output: ContactOutput containing ground reaction forces and torques
            fps: Frame rate for time derivative calculations (default: 20.0)
            com_results: Pre-computed COM results (optional, will compute if None)
            return_components: Whether to return individual loss components (default: False)
            
        Returns:
            If return_components=False: Total weighted loss (scalar tensor)
            If return_components=True: Dictionary with individual loss components and metrics
        """
        # Compute COM dynamics if not provided
        if com_results is None:
            com_results = self.com_calculator.calculate_center_of_mass(
                joints=joints,
                joints_ori=joints_ori,
                fps=fps,
                smooth_derivatives=True
            )
        
        # Extract required quantities
        com_position = com_results["com_position"]  # (B, 3)
        com_acceleration = com_results["com_acceleration"]  # (B, 3)
        moment_about_com = com_results["moment_about_com"]  # (B, 3)
        total_mass = com_results["total_mass"]  # scalar
        
        # Calculate translational consistency loss
        translational_loss = self._calculate_translational_loss(
            com_acceleration=com_acceleration,
            total_mass=total_mass,
            total_force=contact_output.force,
            fps=fps
        )
        
        # Calculate rotational consistency loss
        rotational_loss = self._calculate_rotational_loss(
            com_position=com_position,
            moment_about_com=moment_about_com,
            contact_output=contact_output,
            fps=fps
        )
        
        # Combine losses
        total_loss = (self.translational_weight * translational_loss + 
                     self.rotational_weight * rotational_loss)
        
        if return_components:
            return {
                "total_loss": total_loss,
                "translational_loss": translational_loss,
                "rotational_loss": rotational_loss,
                "com_results": com_results
            }
        else:
            return total_loss
    
    def _calculate_translational_loss(self,
                                    com_acceleration: torch.Tensor,
                                    total_mass: torch.Tensor,
                                    total_force: torch.Tensor,
                                    fps: float) -> torch.Tensor:
        """
        Calculate translational consistency loss: F = ma
        
        Loss: L_trans = ||m*(a - g) - F_GRF||²
        
        Note: gravity_vector points downward, so we subtract it from acceleration
        to get the required external force (GRF) needed to produce the observed acceleration.
        """
        B = com_acceleration.shape[0]
        device = com_acceleration.device
        
        # Expand gravity vector to batch size
        gravity_expanded = self.gravity_vector.unsqueeze(0).expand(B, -1).to(device)
        
        # Calculate required force from Newton's second law: F = m*(a - g)
        # During free fall (flight phase), a ≈ -g, so required force ≈ 0
        required_force = total_mass * (com_acceleration - gravity_expanded)
        
        # Calculate force residual (difference between required and actual GRF)
        force_residual = required_force - total_force
        
        # Apply normalization if requested
        if self.normalize_by_mass:
            force_residual = force_residual / (total_mass + 1e-8)
        
        if self.normalize_by_time:
            # Normalize by time step to make loss independent of frame rate
            dt = 1.0 / fps
            force_residual = force_residual * dt
        
        # Calculate loss
        return self.loss_fn(force_residual, torch.zeros_like(force_residual))
    
    def _calculate_rotational_loss(self,
                                 com_position: torch.Tensor,
                                 moment_about_com: torch.Tensor,
                                 contact_output: ContactOutput,
                                 fps: float) -> torch.Tensor:
        """
        Calculate rotational consistency loss: τ = dL/dt
        
        Loss: L_rot = ||τ_GRF_about_COM - dL/dt||²
        """
        # Calculate moment from ground reaction forces about COM
        grf_moment_about_com = self._calculate_grf_moment_about_com(
            com_position=com_position,
            contact_output=contact_output
        )
        
        # Calculate moment residual (difference between GRF moment and required moment)
        moment_residual = grf_moment_about_com - moment_about_com
        
        # Apply normalization if requested
        if self.normalize_by_time:
            # Normalize by time step to make loss independent of frame rate
            dt = 1.0 / fps
            moment_residual = moment_residual * dt
        
        # Calculate loss
        return self.loss_fn(moment_residual, torch.zeros_like(moment_residual))
    
    def _calculate_grf_moment_about_com(self,
                                      com_position: torch.Tensor,
                                      contact_output: ContactOutput) -> torch.Tensor:
        """
        Calculate the total moment from ground reaction forces about the center of mass.
        
        Moment = r_com_to_cop × F_total + τ_total
        """
        # Vector from COM to center of pressure
        r_com_to_cop = contact_output.cop - com_position  # (B, 3)
        
        # Moment from force at CoP
        moment_from_force = torch.cross(r_com_to_cop, contact_output.force, dim=1)  # (B, 3)
        
        # Total moment about COM
        total_moment = moment_from_force + contact_output.torque  # (B, 3)
        
        return total_moment


def calculate_dynamical_consistency_losses(com_results: Dict[str, torch.Tensor],
                                         contact_output: ContactOutput,
                                         fps: float = 20.0,
                                         gravity: float = 9.81,
                                         translational_weight: float = 1.0,
                                         rotational_weight: float = 1.0,
                                         return_detailed: bool = False,
                                         print_summary: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Simplified function to calculate dynamical consistency losses using pre-computed COM results.
    
    Args:
        com_results: Pre-computed COM analysis results from CenterOfMassCalculator
        contact_output: ContactOutput containing ground reaction forces and torques
        fps: Frame rate for time derivative calculations (default: 20.0)
        gravity: Gravitational acceleration (default: 9.81)
        translational_weight: Weight for translational loss (default: 1.0)
        rotational_weight: Weight for rotational loss (default: 1.0)
        return_detailed: Whether to return detailed analysis (default: False)
        print_summary: Whether to print summary of results (default: False)
        
    Returns:
        If return_detailed=False: Total weighted loss (scalar tensor)
        If return_detailed=True: Dictionary with detailed analysis
    """
    # Extract required quantities from pre-computed COM results
    com_position = com_results["com_position"]  # (B, 3)
    com_acceleration = com_results["com_acceleration"]  # (B, 3)
    moment_about_com = com_results["moment_about_com"]  # (B, 3)
    total_mass = com_results["total_mass"]  # scalar
    
    B = com_position.shape[0]
    device = com_position.device
    
    # Gravity vector (pointing downward in Y direction)
    gravity_vector = torch.tensor([0.0, -gravity, 0.0], device=device)
    gravity_expanded = gravity_vector.unsqueeze(0).expand(B, -1)
    
    # Calculate translational consistency loss: F = ma
    # Required force from Newton's second law: F = m*(a - g)
    required_force = total_mass * (com_acceleration - gravity_expanded)
    force_residual = required_force - contact_output.force
    
    # Normalize by mass and time for better numerical properties
    dt = 1.0 / fps
    force_residual_normalized = (force_residual / (total_mass + 1e-8)) * dt
    translational_loss = torch.mean(force_residual_normalized ** 2)
    
    # Calculate rotational consistency loss: τ = dL/dt
    # Calculate moment from ground reaction forces about COM
    r_com_to_cop = contact_output.cop - com_position  # (B, 3)
    
    # Handle NaN values in CoP (flight phases)
    valid_cop_mask = ~torch.isnan(r_com_to_cop).any(dim=1)  # (B,)
    
    # Initialize GRF moment about COM
    grf_moment_about_com = torch.zeros_like(moment_about_com)
    
    if valid_cop_mask.any():
        # Only calculate moments for frames with valid CoP
        valid_r = r_com_to_cop[valid_cop_mask]  # (N_valid, 3)
        valid_force = contact_output.force[valid_cop_mask]  # (N_valid, 3)
        valid_torque = contact_output.torque[valid_cop_mask]  # (N_valid, 3)
        
        # Moment from force at CoP + direct torque
        moment_from_force = torch.cross(valid_r, valid_force, dim=1)  # (N_valid, 3)
        grf_moment_about_com[valid_cop_mask] = moment_from_force + valid_torque
    
    # Calculate moment residual
    moment_residual = grf_moment_about_com - moment_about_com
    moment_residual_normalized = moment_residual * dt
    rotational_loss = torch.mean(moment_residual_normalized ** 2)
    
    # Combine losses
    total_loss = translational_weight * translational_loss + rotational_weight * rotational_loss
    
    if print_summary:
        print(f"Dynamical Consistency Loss: {total_loss:.6f}")
        if return_detailed:
            print(f"  Translational loss: {translational_loss:.6f}")
            print(f"  Rotational loss: {rotational_loss:.6f}")
            print(f"  Valid CoP frames: {valid_cop_mask.sum().item()}/{B}")
    
    if return_detailed:
        return {
            "total_loss": total_loss,
            "translational_loss": translational_loss,
            "rotational_loss": rotational_loss,
            "force_residual": force_residual,
            "moment_residual": moment_residual,
            "required_force": required_force,
            "grf_moment_about_com": grf_moment_about_com,
            "valid_cop_frames": valid_cop_mask.sum().item(),
            "total_frames": B
        }
    else:
        return total_loss 