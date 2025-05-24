#!/usr/bin/env python
"""
Visualization utilities for plotting joints and creating 3D visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from typing import List, Tuple, Optional

class JointVisualizer:
    """
    Class for visualizing joints and creating 3D visualizations.
    """
    
    @staticmethod
    def plot_joints(
        smpl_joints: np.ndarray, 
        opensim_joints: np.ndarray, 
        save_path: str,
        create_unlabeled: bool = False
    ) -> None:
        """
        Plot SMPL and OpenSim joints overlayed in 3D.
        
        Args:
            smpl_joints (np.ndarray): SMPL joints with shape (J, 3)
            opensim_joints (np.ndarray): OpenSim joints with shape (J, 3)
            save_path (str): Path to save the visualization
            create_unlabeled (bool, optional): Whether to create the unlabeled plot. Defaults to False.
        """
        # Calculate view ranges for consistent plotting
        max_range = np.array([
            np.max([smpl_joints[:, 0].max(), opensim_joints[:, 0].max()]) - 
            np.min([smpl_joints[:, 0].min(), opensim_joints[:, 0].min()]),
            np.max([smpl_joints[:, 1].max(), opensim_joints[:, 1].max()]) - 
            np.min([smpl_joints[:, 1].min(), opensim_joints[:, 1].min()]),
            np.max([smpl_joints[:, 2].max(), opensim_joints[:, 2].max()]) - 
            np.min([smpl_joints[:, 2].min(), opensim_joints[:, 2].min()])
        ]).max() / 2.0
        
        mid_x = (np.max([smpl_joints[:, 0].max(), opensim_joints[:, 0].max()]) + 
                np.min([smpl_joints[:, 0].min(), opensim_joints[:, 0].min()])) / 2
        mid_y = (np.max([smpl_joints[:, 1].max(), opensim_joints[:, 1].max()]) + 
                np.min([smpl_joints[:, 1].min(), opensim_joints[:, 1].min()])) / 2
        mid_z = (np.max([smpl_joints[:, 2].max(), opensim_joints[:, 2].max()]) + 
                np.min([smpl_joints[:, 2].min(), opensim_joints[:, 2].min()])) / 2
                
        # Only create the unlabeled plot if requested
        if create_unlabeled:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot SMPL joints
            ax.scatter(
                smpl_joints[:, 0], 
                smpl_joints[:, 1], 
                smpl_joints[:, 2], 
                c='blue', 
                marker='o', 
                s=50, 
                label='SMPL Joints'
            )
            
            # Plot OpenSim joints
            ax.scatter(
                opensim_joints[:, 0], 
                opensim_joints[:, 1], 
                opensim_joints[:, 2], 
                c='red', 
                marker='x', 
                s=50, 
                label='OpenSim Joints'
            )
            
            # Add labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('SMPL vs OpenSim Joints')
            ax.legend()
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Saved joint visualization to {save_path}")
        
        # Create a plot with labeled joints (always created)
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot SMPL joints
        ax.scatter(
            smpl_joints[:, 0], 
            smpl_joints[:, 1], 
            smpl_joints[:, 2], 
            c='blue', 
            marker='o', 
            s=30, 
            label='SMPL Joints'
        )
        
        # Plot OpenSim joints
        ax.scatter(
            opensim_joints[:, 0], 
            opensim_joints[:, 1], 
            opensim_joints[:, 2], 
            c='red', 
            marker='x', 
            s=30, 
            label='OpenSim Joints'
        )
        
        # Add labels for joint indices
        for i in range(opensim_joints.shape[0]):
            ax.text(
                opensim_joints[i, 0], 
                opensim_joints[i, 1], 
                opensim_joints[i, 2], 
                f"{i}", 
                color='black', 
                fontsize=8
            )
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SMPL vs OpenSim Joints (Labeled)')
        ax.legend()
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save the labeled figure
        plt.tight_layout()
        labeled_path = save_path.replace('.png', '_labeled.png') if create_unlabeled else save_path
        plt.savefig(labeled_path)
        print(f"Saved labeled joint visualization to {labeled_path}")
        
        plt.close('all')
    
    @staticmethod
    def create_joints_mesh(
        opensim_joints: np.ndarray, 
        smpl_joints: np.ndarray, 
        save_path: str
    ) -> None:
        """
        Create a 3D visualization of joints as spheres.
        
        Args:
            opensim_joints (np.ndarray): OpenSim joints
            smpl_joints (np.ndarray): SMPL joints
            save_path (str): Path to save the output mesh
        """
        # Create spheres at each joint position for better visualization
        sphere_radius = 0.02
        sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        
        # Create OpenSim joint spheres (red)
        opensim_meshes = []
        for i in range(opensim_joints.shape[0]):
            joint_pos = opensim_joints[i]
            joint_mesh = sphere_mesh.copy()
            joint_mesh.visual.face_colors = [255, 0, 0, 255]  # Red
            joint_mesh.apply_translation(joint_pos)
            opensim_meshes.append(joint_mesh)
        
        # Create SMPL joint spheres (blue)
        smpl_meshes = []
        for i in range(smpl_joints.shape[0]):
            joint_pos = smpl_joints[i]
            joint_mesh = sphere_mesh.copy()
            joint_mesh.visual.face_colors = [0, 0, 255, 255]  # Blue
            joint_mesh.apply_translation(joint_pos)
            smpl_meshes.append(joint_mesh)
        
        # Combine all spheres
        all_meshes = opensim_meshes + smpl_meshes
        joints_vis = trimesh.util.concatenate(all_meshes)
        
        # Save as OBJ
        joints_vis.export(save_path)
        print(f"Saved 3D visualization mesh to {save_path}") 