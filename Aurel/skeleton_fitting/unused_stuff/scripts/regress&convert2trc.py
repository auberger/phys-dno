#!/usr/bin/env python
"""
Main script for regressing joint locations and creating TRC files.
This provides a clean interface to the underlying functionality.
"""

import os
import torch
import pickle
import numpy as np
from skel.skel_model import SKEL
from joints_utils import JointUtils
from visualization import JointVisualizer
from calc_dist import calculate_joint_distances, save_distances_to_json

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
        Regress joint locations using the SKEL model or SMPL model if gender is neutral.
        
        Args:
            gender (str, optional): Gender for the model. Defaults to "neutral".
            create_visualizations (bool, optional): Whether to create visualizations. Defaults to True.
            rotate_output (bool, optional): Whether to rotate output joints 90 degrees around Y. Defaults to True.
            num_frames (int, optional): Number of frames to write to TRC. Defaults to 10.
            data_rate (float, optional): Data rate in Hz. Defaults to 60.0.
        """
        print(f"Using model with gender: {gender}")
        print(f"Using device: {self.device}")
        
        if gender == "neutral":
            # Use SMPL model for neutral gender
            np_opensim_joints, np_smpl_joints, skin_v0 = self._use_smpl_model()
        else:
            # Use SKEL model for male/female
            np_opensim_joints, np_smpl_joints, skin_v0 = self._use_skel_model(gender)
        
        print(f"OpenSim joints shape: {np_opensim_joints.shape}")
        print(f"SMPL joints shape: {np_smpl_joints.shape}")
        
        # Save the joints as numpy array
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
            
            # Get faces for visualization
            if gender == "neutral":
                # Load SMPL faces
                smpl_faces = np.load('body_models/smpl/smplfaces.npy')
            else:
                # Get SKEL faces
                skel_model = SKEL(gender=gender).to(self.device)
                smpl_faces = skel_model.skin_f.cpu().numpy()
            
            # Save template mesh for reference
            JointUtils.save_mesh(
                skin_v0,
                smpl_faces,
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
        
        # Calculate segment lengths and save to JSON
        print("\nCalculating segment lengths from regressed joints...")
        
        # Reshape joints to have a single frame for calculate_joint_distances
        # Add batch dimension (frames) to the joints
        reshaped_smpl_joints = np_smpl_joints.reshape(1, np_smpl_joints.shape[0], 3)
        
        # Calculate joint distances
        joint_distances_df = calculate_joint_distances(reshaped_smpl_joints)
        print(f"Calculated distances for {len(joint_distances_df)} frames")
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, "regressed_SMPL_distances.json")
        osim_csv_path = "setup/distances_osim.csv"  # Path to the OpenSim distances CSV file if available
        
        # Create a dummy filename based on gender
        filename = f"regressed_{gender}"
        
        save_distances_to_json(
            joint_distances_df,
            filename,
            json_path=json_path,
            osim_csv_path=osim_csv_path
        )
        
        print(f"Saved segment lengths to {json_path}")
        
        print("\nJoint regression complete!")
    
    def _use_skel_model(self, gender):
        """
        Use the SKEL model to regress joints.
        
        Args:
            gender (str): Gender for the SKEL model.
            
        Returns:
            tuple: (opensim_joints, smpl_joints, vertices) as numpy arrays
        """
        print(f"Using SKEL model with gender: {gender}")
        
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
        
        # Print joint names
        print("\nOpenSim joint names:")
        for i, name in enumerate(skel_model.joints_name):
            print(f"{i}: {name}")
            
        return opensim_joints.detach().cpu().numpy()[0], smpl_joints.detach().cpu().numpy()[0], skin_v0[0].detach().cpu().numpy()
    
    def _use_smpl_model(self):
        """
        Use the SMPL neutral model to regress joints.
        
        Returns:
            tuple: (opensim_joints, smpl_joints, vertices) as numpy arrays
        """
        print("Using SMPL neutral model")
        
        # Load SMPL model
        smpl_path = 'body_models/smpl/SMPL_NEUTRAL.pkl'
        with open(smpl_path, 'rb') as f:
            smpl_model = pickle.load(f, encoding='latin1')
        
        print("SMPL model loaded successfully")
        
        # Get template vertices (zero pose, zero shape)
        v_template = smpl_model['v_template']
        v_template_tensor = torch.FloatTensor(v_template).unsqueeze(0).to(self.device)
        
        print(f"SMPL template vertices shape: {v_template.shape}")
        
        # Load SKEL model to get OpenSim regressor
        # Use female model just to get the regressor
        skel_model = SKEL(gender="female").to(self.device)
        
        # Regress OpenSim joint locations using the SKEL regressor
        opensim_joints = torch.einsum("bik,ji->bjk", [v_template_tensor, skel_model.J_regressor_osim])
        
        # Use SMPL's built-in joint regressor
        # Get the joint regressor from the SMPL model
        j_regressor = smpl_model['J_regressor']
        if isinstance(j_regressor, np.ndarray):
            j_regressor_tensor = torch.FloatTensor(j_regressor).to(self.device)
        else:
            # It might be a scipy sparse matrix, convert to dense
            j_regressor_tensor = torch.FloatTensor(j_regressor.todense()).to(self.device)
        
        # Regress SMPL joints
        smpl_joints = torch.matmul(j_regressor_tensor, v_template_tensor[0])
        
        print(f"Regressed SMPL joints shape: {smpl_joints.shape}")
        
        return opensim_joints.detach().cpu().numpy()[0], smpl_joints.detach().cpu().numpy(), v_template
        
    def convert_npy_to_trc(
        self,
        npy_path: str,
        type: str = "smpl",
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

        JointUtils.write_trc_from_npy(
            npy_path=npy_path,
            save_path=save_path,
            type=type,
            data_rate=data_rate,
            rotate=rotate_output
        )

        print(f"Converted {npy_path} to {save_path}")

if __name__ == "__main__":
    # Create regressor
    regressor = JointRegressor(output_dir="Aurel/skeleton_fitting/output")

    ##################### Get Template model (beta=0, theta=0) and regress SKEL joints and SMPL joints #####################

    # Regress SKEL joints from SMPL joints (male gender)
    regressor.regress_skel_joints(
        gender="male",
        create_visualizations=True,
        rotate_output=True,
        num_frames=10,
        data_rate=60.0
    )
    breakpoint()
    ##################### Convert motion data from NPY to TRC #####################

    # Convert motion data from NPY to TRC (only for visualization purposes)
    regressor.convert_npy_to_trc(
        npy_path="Aurel/skeleton_fitting/dno_example_output_save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/results.npy",
        output_name="regressor/results",
        type="smpl",
        data_rate=30.0,
        rotate_output=False
    ) 

    regressor.convert_npy_to_trc(
        npy_path="Aurel/skeleton_fitting/output/regressor/regressed_joints.npy",
        output_name="regressor/regressed_joints",
        type="skel",
        data_rate=30.0,
        rotate_output=False
    ) 