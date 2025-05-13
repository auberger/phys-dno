#!/usr/bin/env python
"""
Main script for regressing joint locations and creating TRC files.
This provides a clean interface to the underlying functionality.
"""

import os
import torch
import numpy as np
from skel.skel_model import SKEL
from utils.joints_utils import JointUtils
from utils.visualization import JointVisualizer

class JointRegressor:
    """
    Main class for regressing joint locations and creating TRC files.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the JointRegressor.
        
        Args:
            output_dir (str, optional): Directory to save output files. Defaults to "output".
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def regress_skel_joints(
        self, 
        gender: str = "neutral", 
        create_visualizations: bool = True,
        rotate_output: bool = True,
        num_frames: int = 10,
        data_rate: float = 60.0
    ) -> None:
        """
        Regress joint locations using the SKEL model.
        
        Args:
            gender (str, optional): Gender for the SKEL model. Defaults to "neutral".
            create_visualizations (bool, optional): Whether to create visualizations. Defaults to True.
            rotate_output (bool, optional): Whether to rotate output joints 90 degrees around Y. Defaults to True.
            num_frames (int, optional): Number of frames to write to TRC. Defaults to 10.
            data_rate (float, optional): Data rate in Hz. Defaults to 60.0.
        """
        print(f"Regressing joints using SKEL model with gender: {gender}")
        print(f"Using device: {self.device}")
        
        # Initialize SKEL model
        skel_model = SKEL(gender=gender).to(self.device)
        print("SKEL model initialized successfully")
        
        # Get the template vertices (equivalent to SMPL with beta=0, theta=0)
        skin_v0 = skel_model.skin_template_v.unsqueeze(0)  # Add batch dimension
        
        print(f"Template vertices shape: {skin_v0.shape}")
        
        # Regress OpenSim joint locations using the SKEL regressor
        opensim_joints = torch.einsum("bik,ji->bjk", [skin_v0, skel_model.J_regressor_osim])
        
        # Convert sparse J_regressor to dense for use with einsum
        dense_j_regressor = skel_model.J_regressor.to_dense()
        
        # Also regress regular SMPL joints for comparison
        smpl_joints = torch.einsum("bik,ji->bjk", [skin_v0, dense_j_regressor])
        
        print(f"OpenSim joints shape: {opensim_joints.shape}")
        print(f"Number of OpenSim joints: {opensim_joints.shape[1]}")
        print(f"SMPL joints shape: {smpl_joints.shape}")
        
        # Print joint names
        print("\nOpenSim joint names:")
        for i, name in enumerate(skel_model.joints_name):
            print(f"{i}: {name}")
        
        # Save the joints as numpy array
        np_opensim_joints = opensim_joints.detach().cpu().numpy()[0]  # Remove batch dimension
        np_smpl_joints = smpl_joints.detach().cpu().numpy()[0]  # Remove batch dimension
        
        np.save(os.path.join(self.output_dir, f"opensim_joints_{gender}.npy"), np_opensim_joints)
        np.save(os.path.join(self.output_dir, f"smpl_joints_{gender}.npy"), np_smpl_joints)
        print(f"Saved joints to {self.output_dir}/opensim_joints_{gender}.npy")
        print(f"Saved joints to {self.output_dir}/smpl_joints_{gender}.npy")
        
        # Create visualization of joints
        if create_visualizations:
            JointVisualizer.plot_joints(
                np_smpl_joints, 
                np_opensim_joints, 
                os.path.join(self.output_dir, f"joints_comparison_{gender}.png"),
                create_unlabeled=False
            )
            
            # Create 3D visualization
            JointVisualizer.create_joints_mesh(
                np_opensim_joints,
                np_smpl_joints,
                os.path.join(self.output_dir, f"joints_visualization_{gender}.obj")
            )
            
            # Save template mesh for reference
            JointUtils.save_mesh(
                skin_v0[0].detach().cpu().numpy(),
                skel_model.skin_f.cpu().numpy(),
                os.path.join(self.output_dir, f"template_mesh_{gender}.obj")
            )
            print(f"Saved template mesh to {self.output_dir}/template_mesh_{gender}.obj")
        
        # Save joints as TRC file
        JointUtils.write_trc_file(
            np_opensim_joints,
            np_smpl_joints,
            os.path.join(self.output_dir, f"joints_{gender}.trc"),
            num_frames=num_frames,
            data_rate=data_rate,
            rotate=rotate_output
        )
        
        print("\nJoint regression complete!")
    
    def convert_npy_to_trc(
        self,
        npy_path: str,
        output_name: str = "smpl_motion",
        data_rate: float = 30.0,
        rotate_output: bool = False
    ) -> None:
        """
        Convert NPY motion data to TRC format.
        
        Args:
            npy_path (str): Path to the NPY file
            output_name (str, optional): Base name for output TRC file. Defaults to "smpl_motion".
            data_rate (float, optional): Data rate in Hz. Defaults to 30.0.
            rotate_output (bool, optional): Whether to rotate output joints 90 degrees around Y. Defaults to False.
        """
        save_path = os.path.join(self.output_dir, f"{output_name}.trc")
        JointUtils.write_smpl_trc_from_npy(
            npy_path=npy_path,
            save_path=save_path,
            data_rate=data_rate,
            rotate=rotate_output
        )
        print(f"Converted {npy_path} to {save_path}")
