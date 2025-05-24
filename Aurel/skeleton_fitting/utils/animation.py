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

    try:
        data = data.item()["motion"][0]  # Extract the first motion sequence
        
        # Transpose data from (22, 3, 120) to (120, 22, 3)
        data = np.transpose(data, (2, 0, 1))
        print("Data shape:", data.shape)
    except:
        data = data
    
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
    
    # Create line objects for each bone
    lines = []
    for _ in range(len(kinematic_tree)):
        line, = ax.plot([], [], [], linewidth=2, color='k')
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
        
        # Update each bone
        for i, (start, end) in enumerate(kinematic_tree):
            xs = [frame_data[start, 0], frame_data[end, 0]]
            ys = [frame_data[start, 2], frame_data[end, 2]]
            zs = [frame_data[start, 1], frame_data[end, 1]]
            
            # Update line data
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
        
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
        kinematic_tree: List of bones (start_joint, end_joint)
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
    
    # Plot bones
    for (start, end) in kinematic_tree:
        xs = [frame_data[start, 0], frame_data[end, 0]]
        ys = [frame_data[start, 2], frame_data[end, 2]]
        zs = [frame_data[start, 1], frame_data[end, 1]]
        ax.plot(xs, ys, zs, linewidth=2, color='k')
    
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
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show plot
    plt.show()

def visualize_with_contact_forces(joints, joints_ori, trans, contact_output, frame_idx=0, force_scale=100.0, save_path=None):
    """
    Visualize the skeleton with contact spheres and ground reaction forces.
    
    Args:
        joints: Joint positions (num_frames, num_joints, 3)
        joints_ori: Joint orientations (num_frames, num_joints, 3, 3)
        trans: Global translation (num_frames, 3)
        contact_output: ContactOutput object containing forces and sphere positions
        frame_idx: Frame index to visualize
        force_scale: Scale factor for force vectors (to make them visible)
        save_path: Path to save the visualization (if None, will display instead)
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Get frame data
    frame_joints = joints[frame_idx]
    frame_trans = trans[frame_idx]
    
    # Define the kinematic tree as a list of bones (start_joint, end_joint)
    kinematic_tree = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Right leg
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),       # Left leg
        (0, 11), (11, 12), (12, 13),                   # Spine to head
        (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # Right arm
        (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # Left arm
    ]
    
    # Plot bones
    for (start, end) in kinematic_tree:
        xs = [frame_joints[start, 0], frame_joints[end, 0]]
        ys = [frame_joints[start, 2], frame_joints[end, 2]]
        zs = [frame_joints[start, 1], frame_joints[end, 1]]
        ax.plot(xs, ys, zs, linewidth=2, color='k')
    
    # Plot contact spheres
    sphere_positions = contact_output.sphere_positions[frame_idx].cpu().numpy()
    sphere_forces = contact_output.sphere_forces[frame_idx].cpu().numpy()
    
    # Plot spheres for right foot (first 6 spheres)
    for i in range(6):
        pos = sphere_positions[i]
        force = sphere_forces[i]
        # Plot sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 0.032 * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = 0.032 * np.outer(np.sin(u), np.sin(v)) + pos[2]  # Z becomes Y
        z = 0.032 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[1]  # Y becomes Z
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        # Plot force vector if force is significant
        if np.linalg.norm(force) > 1e-6:
            force_vec = force * force_scale
            ax.quiver(pos[0], pos[2], pos[1], 
                     force_vec[0], force_vec[2], force_vec[1],
                     color='red', alpha=0.7)
    
    # Plot spheres for left foot (last 6 spheres)
    for i in range(6, 12):
        pos = sphere_positions[i]
        force = sphere_forces[i]
        # Plot sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 0.032 * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = 0.032 * np.outer(np.sin(u), np.sin(v)) + pos[2]  # Z becomes Y
        z = 0.032 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[1]  # Y becomes Z
        ax.plot_surface(x, y, z, color='green', alpha=0.3)
        
        # Plot force vector if force is significant
        if np.linalg.norm(force) > 1e-6:
            force_vec = force * force_scale
            ax.quiver(pos[0], pos[2], pos[1], 
                     force_vec[0], force_vec[2], force_vec[1],
                     color='red', alpha=0.7)
    
    # Plot total GRF for each foot
    force_right = contact_output.force_right[frame_idx].cpu().numpy()
    force_left = contact_output.force_left[frame_idx].cpu().numpy()
    cop_right = contact_output.cop_right[frame_idx].cpu().numpy()
    cop_left = contact_output.cop_left[frame_idx].cpu().numpy()
    
    # Plot right foot GRF
    if np.linalg.norm(force_right) > 1e-6:
        force_vec = force_right * force_scale
        ax.quiver(cop_right[0], cop_right[2], cop_right[1],
                 force_vec[0], force_vec[2], force_vec[1],
                 color='blue', alpha=0.7, label='Right GRF')
    
    # Plot left foot GRF
    if np.linalg.norm(force_left) > 1e-6:
        force_vec = force_left * force_scale
        ax.quiver(cop_left[0], cop_left[2], cop_left[1],
                 force_vec[0], force_vec[2], force_vec[1],
                 color='green', alpha=0.7, label='Left GRF')
    
    # Plot ground plane
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Ground at y=0
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Z (lateral)")
    ax.set_zlabel("Y (up)")
    ax.set_title(f"Frame {frame_idx} - Contact Forces (scale: {force_scale})")
    
    # Set view angle
    ax.view_init(elev=15, azim=-65)
    
    # Add legend
    plt.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()

def animate_with_contact_forces(joints, joints_ori, trans, contact_output, fps=20, force_scale=100.0, save_path=None):
    """
    Create an animation of the skeleton with contact spheres and ground reaction forces.
    
    Args:
        joints: Joint positions (num_frames, num_joints, 3)
        joints_ori: Joint orientations (num_frames, num_joints, 3, 3)
        trans: Global translation (num_frames, 3)
        contact_output: ContactOutput object containing forces and sphere positions
        fps: Frames per second for the animation
        force_scale: Scale factor for force vectors (to make them visible)
        save_path: Path to save the animation (if None, will display instead)
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Define the kinematic tree as a list of bones (start_joint, end_joint)
    kinematic_tree = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Right leg
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),       # Left leg
        (0, 11), (11, 12), (12, 13),                   # Spine to head
        (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # Right arm
        (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # Left arm
    ]
    
    # Get global min/max for consistent axis limits
    all_data = joints.cpu().numpy()  # Convert to numpy for min/max calculation
    margin = 0.5
    global_xmin, global_ymin, global_zmin = all_data.min(axis=(0, 1)) - margin
    global_xmax, global_ymax, global_zmax = all_data.max(axis=(0, 1)) + margin
    
    # Create line objects for each bone
    lines = []
    for _ in range(len(kinematic_tree)):
        line, = ax.plot([], [], [], linewidth=2, color='k')
        lines.append(line)
    
    # Create scatter object for all joints
    scatter = ax.scatter([], [], [], color='black', s=30)
    
    # Create sphere surfaces (will be updated in animation) with fewer points for better performance
    sphere_surfaces = []
    for i in range(12):  # 12 spheres total
        # Reduced resolution for better performance
        u = np.linspace(0, 2 * np.pi, 10)  # Reduced from 20 to 10
        v = np.linspace(0, np.pi, 10)      # Reduced from 20 to 10
        x = 0.032 * np.outer(np.cos(u), np.sin(v))
        y = 0.032 * np.outer(np.sin(u), np.sin(v))
        z = 0.032 * np.outer(np.ones(np.size(u)), np.cos(v))
        surface = ax.plot_surface(x, y, z, alpha=0.3, color='blue' if i < 6 else 'green')
        sphere_surfaces.append(surface)
    
    # Create quiver objects for total GRF (these will be updated in each frame)
    grf_right = None
    grf_left = None
    
    # Plot ground plane
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Ground at y=0
    ground = ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)
    
    def init():
        # Set fixed bounds for the axes
        ax.set_xlim([global_xmin, global_xmax])
        ax.set_ylim([global_zmin, global_zmax])
        ax.set_zlim([global_ymin, global_ymax])
        
        # Set labels and title
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Z (lateral)")
        ax.set_zlabel("Y (up)")
        
        # Enable grid
        ax.grid(True)
        
        # Set view angle
        ax.view_init(elev=15, azim=-65)
        
        # Add legend only once
        blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7)
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7)
        ax.legend([blue_proxy, green_proxy], ['Right GRF', 'Left GRF'])
        
        return lines + [scatter] + sphere_surfaces + [ground]
    
    def update(frame):
        nonlocal grf_right, grf_left
        
        # Get frame data
        frame_joints = joints[frame].cpu().numpy()  # Convert to numpy
        frame_trans = trans[frame].cpu().numpy()  # Convert to numpy
        
        # Update skeleton
        for i, (start, end) in enumerate(kinematic_tree):
            xs = [frame_joints[start, 0], frame_joints[end, 0]]
            ys = [frame_joints[start, 2], frame_joints[end, 2]]
            zs = [frame_joints[start, 1], frame_joints[end, 1]]
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
        
        # Update scatter points
        scatter._offsets3d = (frame_joints[:, 0], frame_joints[:, 2], frame_joints[:, 1])
        
        # Update spheres
        sphere_positions = contact_output.sphere_positions[frame].cpu().numpy()
        
        for i in range(12):
            pos = sphere_positions[i]
            
            # Update sphere position - reduced resolution for better performance
            u = np.linspace(0, 2 * np.pi, 10)  # Reduced from 20 to 10
            v = np.linspace(0, np.pi, 10)      # Reduced from 20 to 10
            x = 0.032 * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = 0.032 * np.outer(np.sin(u), np.sin(v)) + pos[2]
            z = 0.032 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[1]
            
            # Remove old surface and create new one
            sphere_surfaces[i].remove()
            sphere_surfaces[i] = ax.plot_surface(x, y, z, alpha=0.3, 
                                               color='blue' if i < 6 else 'green')
        
        # Update total GRF
        force_right = contact_output.force_right[frame].cpu().numpy()
        force_left = contact_output.force_left[frame].cpu().numpy()
        cop_right = contact_output.cop_right[frame].cpu().numpy()
        cop_left = contact_output.cop_left[frame].cpu().numpy()
        
        # Remove old GRF vectors if they exist
        if grf_right is not None:
            grf_right.remove()
        if grf_left is not None:
            grf_left.remove()
        
        # Update right foot GRF
        if np.linalg.norm(force_right) > 1e-6:
            force_vec = force_right * force_scale
            grf_right = ax.quiver(cop_right[0], cop_right[2], cop_right[1],
                                force_vec[0], force_vec[2], force_vec[1],
                                color='blue', alpha=0.7)
        else:
            grf_right = None
        
        # Update left foot GRF
        if np.linalg.norm(force_left) > 1e-6:
            force_vec = force_left * force_scale
            grf_left = ax.quiver(cop_left[0], cop_left[2], cop_left[1],
                               force_vec[0], force_vec[2], force_vec[1],
                               color='green', alpha=0.7)
        else:
            grf_left = None
        
        # Update title
        ax.set_title(f"Frame {frame}/{len(joints)-1} - Contact Forces (scale: {force_scale})")
        
        artists = lines + [scatter] + sphere_surfaces + [ground]
        if grf_right is not None:
            artists.append(grf_right)
        if grf_left is not None:
            artists.append(grf_left)
        return artists
    
    # Create animation with reduced interval for faster display
    ani = FuncAnimation(
        fig, update, frames=range(len(joints)),
        init_func=init, blit=False, interval=1000/(fps*2))  # Halve the interval to request faster rendering
    
    # Save or show
    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        plt.show()
    
    plt.close()
    
    return ani

def run_animation_workflow(npy_file, fps=100, create_static_plot=False, 
                          save_distances=False, json_path="Aurel/skeleton_fitting/output/SMPL_distances.json", 
                          osim_csv_path="Aurel/skeleton_fitting/setup/distances_osim.csv"):
    """
    Run the full animation workflow including distance calculations.
    
    Args:
        npy_file: Path to the NPY file containing motion data
        fps: Frames per second for the animation
        create_static_plot: Whether to create a static plot with joint annotations
        save_distances: Whether to save distance data to JSON
        json_path: Path to save the JSON file
        osim_csv_path: Path to the OpenSim distances CSV file
        
    Returns:
        tuple: (motion_data, joint_distances_df, distance_data)
    """
    # Define the kinematic tree as a list of bones (start_joint, end_joint)
    kinematic_tree = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Right leg
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),       # Left leg
        (0, 11), (11, 12), (12, 13),                   # Spine to head
        (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # Right arm
        (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # Left arm
    ]

    # Load motion data
    motion_data = load_motion_data(npy_file)
    
    # Create static plot if requested
    if create_static_plot:
        save_path = f"skeleton_{npy_file.split('.')[0]}.png"
        plot_static_skeleton_with_annotations(motion_data, kinematic_tree, frame_idx=0, save_path=save_path)
    
    # Run the animation
    basic_animation(kinematic_tree, motion_data, fps=fps)
    
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

    print(f"Processed {npy_file} successfully.")
    
    return motion_data, joint_distances_df, distance_data