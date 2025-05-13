#!/usr/bin/env python
"""
Utility functions for handling joints data, TRC file operations, and joint regression.
"""

import os
import numpy as np
import torch
import trimesh
from typing import Dict, List, Tuple, Optional, Union

class JointUtils:
    """
    Utilities for joint operations, including TRC file handling and joint regression.
    """
    
    @staticmethod
    def get_osim_marker_names() -> Dict[str, int]:
        """
        Get the OpenSim marker names and their indices.
        
        Returns:
            Dict[str, int]: Dictionary mapping marker names to indices
        """
        return {
            "pelvis": 0, "r_hip": 1, "r_knee": 2, "r_ankle": 3, "r_subtalar": 4,
            "r_mtp": 5, "l_hip": 6, "l_knee": 7, "l_ankle": 8, "l_subtalar": 9,
            "l_mtp": 10, "spine": 11, "r_shoulder": 15, "r_elbow_ulna": 16,
            "r_elbow_radius": 17, "r_hand": 18, "l_shoulder": 20,
            "l_elbow_ulna": 21, "l_elbow_radius": 22, "l_hand": 23
        }
    
    @staticmethod
    def get_smpl_marker_names() -> Dict[str, int]:
        """
        Get the SMPL marker names and their indices.
        
        Returns:
            Dict[str, int]: Dictionary mapping marker names to indices
        """
        return {
            "smpl_pelvis": 0, "l_smpl_hip": 1, "r_smpl_hip": 2, "smpl_spine": 3,
            "l_smpl_knee": 4, "r_smpl_knee": 5, "l_smpl_ankle": 7, "r_smpl_ankle": 8,
            "l_smpl_mtp": 10, "r_smpl_mtp": 11, "smpl_head": 15, "l_smpl_shoulder": 16,
            "r_smpl_shoulder": 17, "l_smpl_elbow": 18, "r_smpl_elbow": 19,
            "l_smpl_hand": 20, "r_smpl_hand": 21
        }
    
    @staticmethod
    def compute_shoulders_middle(opensim_joints: np.ndarray) -> np.ndarray:
        """
        Compute the middle point between shoulders.
        
        Args:
            opensim_joints (np.ndarray): OpenSim joints array
            
        Returns:
            np.ndarray: Middle point between shoulders
        """
        marker_names_osim = JointUtils.get_osim_marker_names()
        r_shoulder_idx = marker_names_osim["r_shoulder"]
        l_shoulder_idx = marker_names_osim["l_shoulder"]
        r_shoulder = opensim_joints[r_shoulder_idx]
        l_shoulder = opensim_joints[l_shoulder_idx]
        return (r_shoulder + l_shoulder) / 2.0
    
    @staticmethod
    def rotate_y_90(joint: np.ndarray) -> np.ndarray:
        """
        Rotate a joint 90 degrees around the y-axis.
        
        Args:
            joint (np.ndarray): Joint coordinates [x, y, z]
            
        Returns:
            np.ndarray: Rotated joint coordinates [z, y, -x]
        """
        if joint.shape != (3,):
            raise ValueError("Joint must be a 3-element array.")
        return np.array([joint[2], joint[1], -joint[0]], dtype=float)
    
    @staticmethod
    def write_trc_file(
        opensim_joints: np.ndarray, 
        smpl_joints: np.ndarray, 
        save_path: str,
        num_frames: int = 10,
        data_rate: float = 60.0,
        rotate: bool = True
    ) -> None:
        """
        Write joint data to a TRC file format that can be loaded in OpenSim.
        
        Args:
            opensim_joints (np.ndarray): OpenSim joints with shape (J, 3)
            smpl_joints (np.ndarray): SMPL joints with shape (J, 3)
            save_path (str): Path to save the TRC file
            num_frames (int, optional): Number of frames to write. Defaults to 10.
            data_rate (float, optional): Data rate in Hz. Defaults to 60.0.
            rotate (bool, optional): Whether to rotate joints 90 degrees around Y. Defaults to True.
        """
        # Get marker names
        marker_names_osim = JointUtils.get_osim_marker_names()
        marker_names_smpl = JointUtils.get_smpl_marker_names()
        
        # Compute shoulders_middle
        shoulders_middle = JointUtils.compute_shoulders_middle(opensim_joints)
        
        # Add the virtual marker to the OpenSim marker list
        marker_names_osim_with_virtual = marker_names_osim.copy()
        marker_names_osim_with_virtual["shoulders_middle"] = -1  # -1 as a placeholder
        
        # Create header
        total_markers = len(marker_names_osim_with_virtual) + len(marker_names_smpl)
        header = [
            "PathFileType\t4\t(X/Y/Z)\tjoints.trc",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            f"{data_rate:.1f}\t{data_rate:.1f}\t{num_frames}\t{total_markers}\tm\t{data_rate:.1f}\t1\t{num_frames}",
            "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names_osim_with_virtual.keys()] + 
                                       [f"{name}\t\t" for name in marker_names_smpl.keys()]),
            "\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(total_markers)])
        ]
        
        # Prepare the data row (identical for all frames)
        def get_data_row(frame_num: int, time_val: float) -> List[str]:
            row = [str(frame_num), f"{time_val:.6f}"]
            # Add OpenSim joint positions (including virtual marker at the end)
            for name, idx in marker_names_osim_with_virtual.items():
                if name == "shoulders_middle":
                    joint = shoulders_middle
                else:
                    joint = opensim_joints[idx]
                if rotate:
                    joint = JointUtils.rotate_y_90(joint)
                row.extend([f"{coord:.6f}" for coord in joint])
            # Add SMPL joint positions
            for name, idx in marker_names_smpl.items():
                joint = smpl_joints[idx]
                if rotate:
                    joint = JointUtils.rotate_y_90(joint)
                row.extend([f"{coord:.6f}" for coord in joint])
            return row
        
        # Write to file
        with open(save_path, "w") as f:
            f.write("\n".join(header) + "\n")
            for i in range(num_frames):
                time_val = i / data_rate
                data_row = get_data_row(i + 1, time_val)
                f.write("\t".join(data_row) + "\n")
        
        print(f"Saved TRC file to {save_path}")

    @staticmethod
    def write_smpl_trc_from_npy(
        npy_path: str, 
        save_path: str,
        data_rate: float = 30.0,
        rotate: bool = False
    ) -> None:
        """
        Convert SMPL joint positions from NPY file to TRC format.
        
        Args:
            npy_path (str): Path to the input NPY file containing SMPL joint positions
            save_path (str): Path to save the output TRC file
            data_rate (float, optional): Data rate in Hz. Defaults to 30.0.
            rotate (bool, optional): Whether to rotate joints 90 degrees around Y. Defaults to False.
        """
        # Get SMPL marker names
        marker_names_smpl = JointUtils.get_smpl_marker_names()
        
        # Load the NPY file
        try:
            joint_positions = np.load(npy_path, allow_pickle=True)
            joint_positions = joint_positions.item()['motion'][0]
            # Transpose data from (22, 3, N) to (N, 22, 3)
            joint_positions = np.transpose(joint_positions, (2, 0, 1))
        except Exception as e:
            raise RuntimeError(f"Failed to load NPY file: {str(e)}")
        
        # Number of frames
        num_frames = joint_positions.shape[0]
        
        # Create header
        total_markers = len(marker_names_smpl)
        header = [
            "PathFileType\t4\t(X/Y/Z)\tjoints.trc",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            f"{data_rate:.1f}\t{data_rate:.1f}\t{num_frames}\t{total_markers}\tm\t{data_rate:.1f}\t1\t{num_frames}",
            "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names_smpl.keys()]),
            "\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(total_markers)])
        ]
        
        # Write to file
        with open(save_path, "w") as f:
            # Write header
            f.write("\n".join(header) + "\n")
            
            # Write data rows
            for frame_idx in range(num_frames):
                # Calculate time for this frame
                time_val = frame_idx / data_rate
                
                # Start row with frame number and time
                row = [str(frame_idx + 1), f"{time_val:.6f}"]
                
                # Add joint positions for this frame
                for name, idx in marker_names_smpl.items():
                    joint = joint_positions[frame_idx, idx]
                    if rotate:
                        joint = JointUtils.rotate_y_90(joint)
                    row.extend([f"{coord:.6f}" for coord in joint])
                
                # Write the row
                f.write("\t".join(row) + "\n")
        
        print(f"Saved TRC file to {save_path}")

    @staticmethod
    def save_mesh(vertices: np.ndarray, faces: np.ndarray, save_path: str) -> None:
        """
        Save vertices and faces as a mesh file.
        
        Args:
            vertices (np.ndarray): Mesh vertices
            faces (np.ndarray): Mesh faces
            save_path (str): Path to save the mesh file
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(save_path)
        print(f"Saved mesh to {save_path}") 