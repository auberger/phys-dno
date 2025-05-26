"""Contact Models Module for PyTorch

This module implements a PyTorch-compatible version of the contact model used in 
biomechanical simulations to calculate ground reaction forces (GRF) based on 
kinematics. The implementation allows for backpropagation through the contact 
force calculations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple

class ContactSphere:
    """Class representing a contact sphere attached to a body segment."""
    
    def __init__(self, 
                 name: str,
                 body_idx: int,
                 local_position: torch.Tensor,
                 radius: float):
        """Initialize a contact sphere.
        
        Args:
            name: Name of the sphere
            body_idx: Index of the body segment this sphere is attached to
            local_position: Position of sphere center in body frame (3D tensor)
            radius: Radius of the sphere
        """
        self.name = name
        self.body_idx = body_idx
        self.local_position = local_position
        self.radius = radius

class ContactOutput(NamedTuple):
    """Output of the contact model containing forces, torques, and CoP."""
    force: torch.Tensor  # Total ground reaction force (B x 3)
    torque: torch.Tensor  # Total ground reaction torque (B x 3)
    cop: torch.Tensor  # Center of pressure (B x 3)
    sphere_forces: torch.Tensor  # Individual sphere forces (B x N_spheres x 3)
    sphere_positions: torch.Tensor  # Sphere positions (B x N_spheres x 3)
    # New fields for separate foot forces and torques
    force_right: torch.Tensor  # Right foot ground reaction force (B x 3)
    force_left: torch.Tensor  # Left foot ground reaction force (B x 3)
    torque_right: torch.Tensor  # Right foot ground reaction torque (B x 3)
    torque_left: torch.Tensor  # Left foot ground reaction torque (B x 3)
    cop_right: torch.Tensor  # Right foot center of pressure (B x 3)
    cop_left: torch.Tensor  # Left foot center of pressure (B x 3)

class ContactModel(nn.Module):
    """PyTorch module for calculating ground reaction forces using contact spheres."""
    
    def __init__(self,
                 stiffness: float = 1e5,
                 dissipation: float = 0.2,
                 static_friction: float = 0.8,
                 dynamic_friction: float = 0.8,
                 viscous_friction: float = 0.5,
                 transition_velocity: float = 0.2,
                 ground_height: float = 0.0,
                 dt: float = 1/20.0):  # Default to 20Hz
        """Initialize the contact model."""
        super().__init__()
        
        # Contact parameters
        self.stiffness = stiffness
        self.dissipation = dissipation
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.viscous_friction = viscous_friction
        self.transition_velocity = transition_velocity
        self.ground_height = ground_height
        self.dt = dt
        
        # Ground normal (pointing up)
        self.register_buffer("ground_normal", torch.tensor([0.0, 1.0, 0.0]))
        
        # Initialize contact spheres based on OpenSim model
        self.spheres = self._initialize_spheres()
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters."""
        if self.stiffness <= 0:
            raise ValueError("Stiffness must be positive")
        if self.dissipation < 0:
            raise ValueError("Dissipation must be non-negative")
        if self.static_friction < 0:
            raise ValueError("Static friction must be non-negative")
        if self.dynamic_friction < 0:
            raise ValueError("Dynamic friction must be non-negative")
        if self.viscous_friction < 0:
            raise ValueError("Viscous friction must be non-negative")
        if self.transition_velocity <= 0:
            raise ValueError("Transition velocity must be positive")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
    
    def _initialize_spheres(self) -> List[ContactSphere]:
        """Initialize contact spheres based on OpenSim model definitions."""
        spheres = []
        device = self.ground_normal.device  # Get the device from the ground_normal tensor
        
        # Right foot spheres
        spheres.extend([
            ContactSphere("s1_r", 4, torch.tensor([0.0019011578840796601, -0.01, -0.00382630379623308], device=device), 0.032),
            ContactSphere("s2_r", 4, torch.tensor([0.14838639994206301, -0.01, -0.028713422052654002], device=device), 0.032),
            ContactSphere("s3_r", 4, torch.tensor([0.13300117060705099, -0.01, 0.051636247344956601], device=device), 0.032),
            ContactSphere("s4_r", 4, torch.tensor([0.066234666199163503, -0.01, 0.026364160674169801], device=device), 0.032),
            ContactSphere("s5_r", 5, torch.tensor([0.059999999999999998, -0.01, -0.018760308461917698], device=device), 0.032),
            ContactSphere("s6_r", 5, torch.tensor([0.044999999999999998, -0.01, 0.061856956754965199], device=device), 0.032),
        ])
        
        # Left foot spheres
        spheres.extend([
            ContactSphere("s1_l", 9, torch.tensor([0.0019011578840796601, -0.01, 0.00382630379623308], device=device), 0.032),
            ContactSphere("s2_l", 9, torch.tensor([0.14838639994206301, -0.01, 0.028713422052654002], device=device), 0.032),
            ContactSphere("s3_l", 9, torch.tensor([0.13300117060705099, -0.01, -0.051636247344956601], device=device), 0.032),
            ContactSphere("s4_l", 9, torch.tensor([0.066234666199163503, -0.01, -0.026364160674169801], device=device), 0.032),
            ContactSphere("s5_l", 10, torch.tensor([0.059999999999999998, -0.01, 0.018760308461917698], device=device), 0.032),
            ContactSphere("s6_l", 10, torch.tensor([0.044999999999999998, -0.01, -0.061856956754965199], device=device), 0.032),
        ])
        
        return spheres
    
    def _get_sphere_positions(self, 
                            joints: torch.Tensor,
                            joints_ori: torch.Tensor) -> torch.Tensor:
        """Calculate global positions of all contact spheres.
        
        Args:
            joints: Joint positions (B x 24 x 3)
            joints_ori: Joint orientations (B x 24 x 3 x 3)
            
        Returns:
            Global positions of all spheres (B x N_spheres x 3)
        """
        B = joints.shape[0]
        N_spheres = len(self.spheres)
        device = joints.device
        
        # Input validation
        if joints.shape[1] != 24:
            raise ValueError(f"Expected 24 joints, got {joints.shape[1]}")
        if joints_ori.shape[1] != 24:
            raise ValueError(f"Expected 24 joint orientations, got {joints_ori.shape[1]}")
        
        # Initialize output tensor
        sphere_positions = torch.zeros(B, N_spheres, 3, device=device)
        
        # Transform each sphere to global coordinates
        for i, sphere in enumerate(self.spheres):
            # Get body position and orientation
            body_pos = joints[:, sphere.body_idx]
            body_ori = joints_ori[:, sphere.body_idx]
            
            # Transform sphere position to global coordinates
            local_pos = sphere.local_position.to(device)
            sphere_positions[:, i] = body_pos + torch.matmul(body_ori, local_pos)
            
        return sphere_positions
    
    def _calculate_contact_force(self,
                               sphere_pos: torch.Tensor,
                               sphere_vel: torch.Tensor,
                               sphere_radius: float) -> torch.Tensor:
        """Calculate contact force for a single sphere.
        
        Args:
            sphere_pos: Sphere position (B x 3)
            sphere_vel: Sphere velocity (B x 3)
            sphere_radius: Radius of the sphere
            
        Returns:
            Contact force (B x 3)
        """
        device = sphere_pos.device
        
        # Constants for numerical stability
        eps = torch.tensor(1e-5, device=device)
        eps2 = torch.tensor(1e-16, device=device)
        bv = torch.tensor(50, device=device)
        bd = torch.tensor(300, device=device)
        
        # Calculate indentation relative to ground height
        indentation = self.ground_height - sphere_pos[:, 1]  # y-coordinate is height
        
        # Calculate normal velocity
        vnormal = torch.sum(sphere_vel * self.ground_normal.to(device), dim=1)
        vtangent = sphere_vel - vnormal.unsqueeze(1) * self.ground_normal.to(device)
        indentation_vel = -vnormal
        
        # Calculate stiffness force (Hertz force)
        k = 0.5 * (self.stiffness) ** (2/3)
        fH = (4/3) * k * torch.sqrt(torch.tensor(sphere_radius * k, device=device)) * \
             (torch.sqrt(indentation * indentation + eps)) ** (3/2)
        
        # Calculate dissipation force
        fHd = fH * (1 + 1.5 * self.dissipation * indentation_vel)
        
        # Calculate normal force with smooth transition
        fn = ((0.5 * torch.tanh(bv * (indentation_vel + 1 / (1.5 * self.dissipation))) + 0.5 + eps2) * 
              (0.5 * torch.tanh(bd * indentation) + 0.5 + eps2) * fHd)
        
        # Calculate friction force
        vslip = torch.sqrt(torch.sum(vtangent * vtangent, dim=1) + eps)
        vrel = vslip / self.transition_velocity
        
        ffriction = fn * (torch.minimum(vrel, torch.ones_like(vrel)) * 
                         (self.dynamic_friction + 2 * (self.static_friction - self.dynamic_friction) / 
                          (1 + vrel * vrel)) + self.viscous_friction * vslip)
        
        # Calculate total force
        force = fn.unsqueeze(1) * self.ground_normal.to(device)
        friction_force = ffriction.unsqueeze(1) * (-vtangent) / (vslip.unsqueeze(1) + eps)
        
        return force + friction_force
    
    def _calculate_cop_and_torque(self,
                                sphere_positions: torch.Tensor,
                                sphere_forces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate center of pressure and ground reaction torque for each foot.
        
        Args:
            sphere_positions: Sphere positions (B x N_spheres x 3)
            sphere_forces: Sphere forces (B x N_spheres x 3)
            
        Returns:
            Tuple of (cop, torque, cop_right, cop_left, torque_right, torque_left) where:
            - cop is the total center of pressure (B x 3)
            - torque is the total ground reaction torque (B x 3)
            - cop_right is the right foot center of pressure (B x 3)
            - cop_left is the left foot center of pressure (B x 3)
            - torque_right is the right foot ground reaction torque (B x 3)
            - torque_left is the left foot ground reaction torque (B x 3)
        """
        B = sphere_positions.shape[0]
        device = sphere_positions.device
        
        # Calculate total vertical force
        total_vertical_force = torch.sum(sphere_forces[:, :, 1], dim=1)  # B
        
        # Calculate weighted average of contact points for CoP
        # Only consider points with positive vertical force
        vertical_forces = sphere_forces[:, :, 1]  # B x N_spheres
        valid_contacts = vertical_forces > 0
        
        # Initialize CoP and torque tensors
        cop = torch.zeros(B, 3, device=device)
        torque = torch.zeros(B, 3, device=device)
        cop_right = torch.zeros(B, 3, device=device)
        cop_left = torch.zeros(B, 3, device=device)
        torque_right = torch.zeros(B, 3, device=device)
        torque_left = torch.zeros(B, 3, device=device)
        
        # Calculate CoP and torque for each batch
        for b in range(B):
            if torch.any(valid_contacts[b]):
                # Get valid contact points and their forces
                valid_pos = sphere_positions[b, valid_contacts[b]]  # N_valid x 3
                valid_forces = sphere_forces[b, valid_contacts[b]]  # N_valid x 3
                
                # Calculate total CoP and torque
                weights = valid_forces[:, 1]  # Vertical forces as weights
                total_weight = torch.sum(weights)
                
                if total_weight > 0:  # Only calculate if there's a positive vertical force
                    cop[b, 0] = torch.sum(valid_pos[:, 0] * weights) / total_weight
                    cop[b, 1] = self.ground_height
                    cop[b, 2] = torch.sum(valid_pos[:, 2] * weights) / total_weight
                    
                    # Calculate torque about CoP
                    r = valid_pos - cop[b]  # Vectors from CoP to contact points
                    torque[b] = torch.sum(torch.cross(r, valid_forces), dim=0)
                
                # Separate calculations for right and left feet
                # Right foot spheres are indices 0-5, left foot are 6-11
                right_mask = valid_contacts[b, :6]  # First 6 spheres are right foot
                left_mask = valid_contacts[b, 6:]   # Last 6 spheres are left foot
                
                # Right foot calculations
                if torch.any(right_mask):
                    right_pos = sphere_positions[b, :6][right_mask]
                    right_forces = sphere_forces[b, :6][right_mask]
                    right_weights = right_forces[:, 1]
                    right_total_weight = torch.sum(right_weights)
                    
                    if right_total_weight > 0:
                        cop_right[b, 0] = torch.sum(right_pos[:, 0] * right_weights) / right_total_weight
                        cop_right[b, 1] = self.ground_height
                        cop_right[b, 2] = torch.sum(right_pos[:, 2] * right_weights) / right_total_weight
                        
                        r_right = right_pos - cop_right[b]
                        torque_right[b] = torch.sum(torch.cross(r_right, right_forces), dim=0)
                
                # Left foot calculations
                if torch.any(left_mask):
                    left_pos = sphere_positions[b, 6:][left_mask]
                    left_forces = sphere_forces[b, 6:][left_mask]
                    left_weights = left_forces[:, 1]
                    left_total_weight = torch.sum(left_weights)
                    
                    if left_total_weight > 0:
                        cop_left[b, 0] = torch.sum(left_pos[:, 0] * left_weights) / left_total_weight
                        cop_left[b, 1] = self.ground_height
                        cop_left[b, 2] = torch.sum(left_pos[:, 2] * left_weights) / left_total_weight
                        
                        r_left = left_pos - cop_left[b]
                        torque_left[b] = torch.sum(torch.cross(r_left, left_forces), dim=0)
        
        return cop, torque, cop_right, cop_left, torque_right, torque_left
    
    def forward(self,
                joints: torch.Tensor,
                joints_ori: torch.Tensor,
                joint_velocities: Optional[torch.Tensor] = None) -> ContactOutput:
        """Calculate ground reaction forces, torques, and CoP for all contact spheres.
        
        Args:
            joints: Joint positions (B x 24 x 3)
            joints_ori: Joint orientations (B x 24 x 3 x 3)
            joint_velocities: Joint velocities (B x 24 x 3), optional
            
        Returns:
            ContactOutput containing:
            - force: Total ground reaction force (B x 3)
            - torque: Total ground reaction torque (B x 3)
            - cop: Center of pressure (B x 3)
            - sphere_forces: Individual sphere forces (B x N_spheres x 3)
            - sphere_positions: Sphere positions (B x N_spheres x 3)
            - force_right: Right foot ground reaction force (B x 3)
            - force_left: Left foot ground reaction force (B x 3)
            - torque_right: Right foot ground reaction torque (B x 3)
            - torque_left: Left foot ground reaction torque (B x 3)
            - cop_right: Right foot center of pressure (B x 3)
            - cop_left: Left foot center of pressure (B x 3)
        """
        # Input validation
        if not isinstance(joints, torch.Tensor):
            raise TypeError("joints must be a torch.Tensor")
        if not isinstance(joints_ori, torch.Tensor):
            raise TypeError("joints_ori must be a torch.Tensor")
        if joint_velocities is not None and not isinstance(joint_velocities, torch.Tensor):
            raise TypeError("joint_velocities must be a torch.Tensor")
        
        B = joints.shape[0]
        device = joints.device
        
        # Get sphere positions
        sphere_positions = self._get_sphere_positions(joints, joints_ori)
        
        # Calculate sphere velocities
        if joint_velocities is None:
            # Use finite difference to estimate velocities
            sphere_velocities = torch.zeros_like(sphere_positions)
            if B > 1:  # Only if we have multiple frames
                sphere_velocities[1:] = (sphere_positions[1:] - sphere_positions[:-1]) / self.dt
        else:
            # Transform joint velocities to sphere velocities
            sphere_velocities = torch.zeros_like(sphere_positions)
            for i, sphere in enumerate(self.spheres):
                body_vel = joint_velocities[:, sphere.body_idx]
                body_ori = joints_ori[:, sphere.body_idx]
                sphere_velocities[:, i] = body_vel + torch.matmul(body_ori, sphere.local_position.to(device))
        
        # Calculate contact forces for each sphere
        sphere_forces = torch.zeros_like(sphere_positions)
        for i, sphere in enumerate(self.spheres):
            sphere_forces[:, i] = self._calculate_contact_force(
                sphere_positions[:, i],
                sphere_velocities[:, i],
                sphere.radius
            )
        
        # Calculate total force
        total_force = torch.sum(sphere_forces, dim=1)
        
        # Calculate forces for each foot
        force_right = torch.sum(sphere_forces[:, :6], dim=1)  # First 6 spheres are right foot
        force_left = torch.sum(sphere_forces[:, 6:], dim=1)   # Last 6 spheres are left foot
        
        # Calculate CoP and torques
        cop, torque, cop_right, cop_left, torque_right, torque_left = self._calculate_cop_and_torque(
            sphere_positions, sphere_forces
        )
        
        return ContactOutput(
            force=total_force,
            torque=torque,
            cop=cop,
            sphere_forces=sphere_forces,
            sphere_positions=sphere_positions,
            force_right=force_right,
            force_left=force_left,
            torque_right=torque_right,
            torque_left=torque_left,
            cop_right=cop_right,
            cop_left=cop_left
        ) 