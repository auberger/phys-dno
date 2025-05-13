import numpy as np
import pandas as pd  # Import pandas for DataFrame functionality
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.append(".")  # Add current directory to path
from utils import calc_dist

def load_motion_data(npy_file):
    """
    Load and preprocess motion data from an NPY file.
    
    Args:
        npy_file: Path to the NPY file containing motion data
        
    Returns:
        numpy.ndarray: Processed motion data with shape (seq_len, num_joints, 3)
    """
    # Load and process the npy file
    data = np.load(npy_file, allow_pickle=True)
    data = data.item()["motion"][0]  # Extract the first motion sequence
    
    # Transpose data from (22, 3, 120) to (120, 22, 3)
    data = np.transpose(data, (2, 0, 1))
    print("Data shape:", data.shape)
    
    return data

def basic_animation(kinematic_tree, joints, fps=20):
    """Simple animation with minimal transformations"""
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Get global min/max for consistent axis limits
    all_data = joints
    margin = 0.5
    global_xmin, global_ymin, global_zmin = all_data.min(axis=(0, 1)) - margin
    global_xmax, global_ymax, global_zmax = all_data.max(axis=(0, 1)) + margin
    
    # Print global min/max for debugging
    print("Global min:", global_xmin, global_ymin, global_zmin)
    print("Global max:", global_xmax, global_ymax, global_zmax)
    
    # Set axis labels
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Z (lateral)")
    ax.set_zlabel("Y (up)")
    
    # Set title
    plt.title("Joint Motion Animation")
    
    # Create line objects for each kinematic chain
    lines = []
    for i in range(len(kinematic_tree)):
        line, = ax.plot([], [], [], linewidth=2, marker='o')
        lines.append(line)
    
    # Create scatter object for all joints
    scatter = ax.scatter([], [], [])
    
    # Initialize function for animation
    def init():
        # Set fixed bounds for the axes
        ax.set_xlim([global_xmin, global_xmax])
        ax.set_ylim([global_zmin, global_zmax])
        ax.set_zlim([global_ymin, global_ymax])
        
        # Enable grid
        ax.grid(True)
        
        # Set a better view angle for human skeleton
        ax.view_init(elev=15, azim=-65)
        
        # Return all artists
        return lines + [scatter]
    
    # Update function for animation
    def update(frame):
        # Get frame data
        frame_data = joints[frame]
        
        # Update each kinematic chain
        for i, (chain, line) in enumerate(zip(kinematic_tree, lines)):
            xs = frame_data[chain, 0]
            ys = frame_data[chain, 2]
            zs = frame_data[chain, 1]
            
            # Update line data
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        
        # Update scatter points
        scatter._offsets3d = (frame_data[:, 0], frame_data[:, 2], frame_data[:, 1])
        
        # Display frame number
        ax.set_title(f"Frame {frame}/{len(joints)-1}")
        
        # Return all artists
        return lines + [scatter]
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=range(len(joints)),
        init_func=init, blit=False, interval=1000/fps)
    
    # Show animation
    plt.tight_layout()
    plt.show()
    
    return ani

def plot_static_skeleton_with_annotations(joints, kinematic_tree, frame_idx=0, save_path="skeleton_with_joints.png"):
    """
    Create a static plot of the skeleton with annotated joint numbers.
    
    Args:
        joints: Motion data with shape (seq_len, num_joints, 3)
        kinematic_tree: List of lists defining the kinematic chains
        frame_idx: Index of the frame to plot
        save_path: Path to save the plot image
    """
    # Get frame data
    frame_data = joints[frame_idx]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Set title
    ax.set_title(f"Skeleton with Joint Numbers - Frame {frame_idx}", fontsize=14)
    
    # Set axis labels
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Z (lateral)")
    ax.set_zlabel("Y (up)")
    
    # Get min/max for axis limits
    margin = 0.5
    xmin, ymin, zmin = frame_data.min(axis=0) - margin
    xmax, ymax, zmax = frame_data.max(axis=0) + margin
    
    # Set axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])  # Z-axis becomes Y-axis in the plot
    ax.set_zlim([ymin, ymax])  # Y-axis becomes Z-axis in the plot
    
    # Enable grid
    ax.grid(True)
    
    # Set view angle
    ax.view_init(elev=15, azim=-65)
    
    # Plot kinematic chains
    colors = ["red", "blue", "green", "purple", "orange"]
    for i, chain in enumerate(kinematic_tree):
        # Get coordinates with Y-up orientation
        xs = frame_data[chain, 0]
        ys = frame_data[chain, 2]  # Z becomes Y in plot
        zs = frame_data[chain, 1]  # Y becomes Z in plot
        
        # Plot the chain
        ax.plot(xs, ys, zs, linewidth=2, marker='o', color=colors[i % len(colors)], label=f"Chain {i}")
    
    # Add scatter points for all joints
    ax.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1], color="black", s=30)
    
    # Add joint number annotations
    for i in range(len(frame_data)):
        # Get joint position
        x, y, z = frame_data[i]
        
        # Text coordinates (with Y-up orientation)
        text_x = x
        text_y = z  # Z becomes Y in plot
        text_z = y  # Y becomes Z in plot
        
        # Add text annotation
        ax.text(text_x, text_y, text_z, f"{i}", fontsize=10, color="black", 
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    
    # Add legend
    plt.legend(loc="upper right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show plot
    plt.show()

def run_animation_workflow(npy_file, kinematic_chain, fps=20, create_static_plot=False, 
                          save_distances=False, json_path="Aurel/skeleton_fitting/output/SMPL_distances.json", 
                          osim_csv_path="Aurel/skeleton_fitting/setup/distances_osim.csv"):
    """
    Run the full animation workflow including distance calculations.
    
    Args:
        npy_file: Path to the NPY file containing motion data
        kinematic_chain: List of lists defining the kinematic chains
        fps: Frames per second for the animation
        create_static_plot: Whether to create a static plot with joint annotations
        save_distances: Whether to save distance data to JSON
        json_path: Path to save the JSON file
        osim_csv_path: Path to the OpenSim distances CSV file
        
    Returns:
        tuple: (motion_data, joint_distances_df, distance_data)
    """
    # Load motion data
    motion_data = load_motion_data(npy_file)
    
    # Create static plot if requested
    if create_static_plot:
        save_path = f"skeleton_{npy_file.split('.')[0]}.png"
        plot_static_skeleton_with_annotations(motion_data, kinematic_chain, frame_idx=0, save_path=save_path)
    
    # Run the animation
    basic_animation(kinematic_chain, motion_data, fps=fps)
    
    # Calculate distances
    joint_distances_df = calc_dist.calculate_joint_distances(motion_data)
    
    # Save distances to JSON if requested
    distance_data = None
    if save_distances:
        distance_data = calc_dist.save_distances_to_json(
            joint_distances_df, 
            npy_file, 
            json_path=json_path, 
            osim_csv_path=osim_csv_path
        )
    
    return motion_data, joint_distances_df, distance_data