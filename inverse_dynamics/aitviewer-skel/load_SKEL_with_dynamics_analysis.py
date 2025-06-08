import argparse
import os
import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.skel import SKELSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.arrows import Arrows
from aitviewer.viewer import Viewer

try:
    from skel.skel_model import SKEL
except Exception as e:
    print("Could not import SKEL, make sure you installed the skel repository.")
    raise e

def create_foot_grf_arrows(cop_positions: torch.Tensor, forces: torch.Tensor, 
                          force_scale: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """
    Create total GRF arrows for each foot at CoP positions.
    
    Args:
        cop_positions: (F, 3) tensor of center of pressure positions
        forces: (F, 3) tensor of total forces for the foot
        force_scale: Scale factor to make forces visible
    
    Returns:
        Tuple of (arrow_origins, arrow_tips) as numpy arrays
    """
    F = cop_positions.shape[0]
    
    # Convert to numpy
    cop_np = cop_positions.cpu().numpy()
    forces_np = forces.cpu().numpy()
    
    # Initialize arrow arrays
    arrow_origins = np.zeros((F, 1, 3))
    arrow_tips = np.zeros((F, 1, 3))
    
    for frame in range(F):
        # Check if CoP is valid (not NaN) and force is significant
        if not np.isnan(cop_np[frame]).any():
            force_magnitude = np.linalg.norm(forces_np[frame])
            if force_magnitude > 1.0:  # Only create arrow if force is significant
                arrow_origins[frame, 0] = cop_np[frame]
                arrow_tips[frame, 0] = cop_np[frame] + forces_np[frame] * force_scale
            else:
                # Set to same position (no visible arrow)
                arrow_origins[frame, 0] = cop_np[frame]
                arrow_tips[frame, 0] = cop_np[frame]
        else:
            # For invalid CoP, set arrows to origin (won't be visible)
            arrow_origins[frame, 0] = [0, 0, 0]
            arrow_tips[frame, 0] = [0, 0, 0]
    
    return arrow_origins, arrow_tips

def create_dynamics_arrows(origins: np.ndarray, forces: np.ndarray, 
                          force_scale: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """
    Create arrows for dynamics analysis (total GRF and required force).
    
    Args:
        origins: (F, 3) array of arrow origins
        forces: (F, 3) array of force vectors
        force_scale: Scale factor to make forces visible
    
    Returns:
        Tuple of (arrow_origins, arrow_tips) as numpy arrays
    """
    F = len(origins)
    
    # Initialize arrow arrays
    arrow_origins = np.zeros((F, 1, 3))
    arrow_tips = np.zeros((F, 1, 3))
    
    for frame in range(F):
        force_magnitude = np.linalg.norm(forces[frame])
        
        if force_magnitude > 1.0:  # Only create arrow if force is significant
            arrow_origins[frame, 0] = origins[frame]
            arrow_tips[frame, 0] = origins[frame] + forces[frame] * force_scale
        else:
            # Set to same position (no visible arrow)
            arrow_origins[frame, 0] = origins[frame]
            arrow_tips[frame, 0] = origins[frame]
    
    return arrow_origins, arrow_tips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a SKEL model with comprehensive dynamics analysis.")
    parser.add_argument("-s", "--motion_file", type=str, help="Path to a skel motion file", required=True)
    parser.add_argument("-c", "--contact_file", type=str, help="Path to contact forces file", required=True)
    parser.add_argument("-m", "--com_file", type=str, help="Path to COM analysis file", required=True)
    parser.add_argument("-z", "--z_up", help="Rotate the mesh 90 deg", action="store_true")
    parser.add_argument("--force_scale", type=float, default=0.001, 
                       help="Scale factor for force arrows (default: 0.001)")

    args = parser.parse_args()

    assert os.path.exists(args.motion_file), f"Could not find {args.motion_file}"
    assert os.path.exists(args.contact_file), f"Could not find {args.contact_file}"
    assert os.path.exists(args.com_file), f"Could not find {args.com_file}"
    assert args.motion_file.endswith(".pkl"), "Motion file should be a .pkl file"
    
    # Load SKEL sequence
    skel_seq = SKELSequence.from_file(args.motion_file, name="SKEL", fps_in=120, fps_out=120, z_up=args.z_up)
    
    # Make SKEL model semi-transparent so arrows inside are visible
    skel_seq.material.color = (0.7, 0.7, 0.7, 0.5)  # Semi-transparent grey
    
    cam_pose = np.array([0, 1.2, 4.0])
    
    # Load contact data
    print("Loading contact data...")
    contact_data = torch.load(args.contact_file)
    total_force = contact_data['force']
    force_right = contact_data['force_right']
    force_left = contact_data['force_left']
    cop_total = contact_data['cop']
    cop_right = contact_data['cop_right']
    cop_left = contact_data['cop_left']
    
    # Load COM data
    print("Loading COM analysis data...")
    import json
    with open(args.com_file, 'r') as f:
        com_data = json.load(f)
    
    # Convert COM data to numpy arrays
    com_position = np.array(com_data['com_position'])
    com_acceleration = np.array(com_data['com_acceleration'])
    total_mass = float(com_data['total_mass'])
    
    # Calculate required force: m*(a - g) for all frames
    gravity_vector = np.array([0.0, -9.81, 0.0])  # Gravity pointing down in Y
    required_force = total_mass * (com_acceleration - gravity_vector)  # (num_frames, 3)
    
    # Create total GRF arrows for each foot
    print("Creating right foot total GRF arrows...")
    right_origins, right_tips = create_foot_grf_arrows(cop_right, force_right, args.force_scale)
    
    print("Creating left foot total GRF arrows...")
    left_origins, left_tips = create_foot_grf_arrows(cop_left, force_left, args.force_scale)
    
    # Create GRF arrows for right foot (black, larger and more visible)
    right_grf_arrows = Arrows(
        origins=right_origins,
        tips=right_tips,
        r_base=0.012,  # Increased size
        r_head=0.020,  # Increased size
        color=(0.0, 0.0, 0.0, 1.0),  # Solid black
        name="Right Foot Total GRF"
    )
    
    # Create GRF arrows for left foot (black, larger and more visible)
    left_grf_arrows = Arrows(
        origins=left_origins,
        tips=left_tips,
        r_base=0.012,  # Increased size
        r_head=0.020,  # Increased size
        color=(0.0, 0.0, 0.0, 1.0),  # Solid black
        name="Left Foot Total GRF"
    )
    
    # Create Center of Mass visualization (single moving sphere)
    com_spheres = Spheres(
        positions=com_position.reshape(-1, 1, 3),  # (F, 1, 3) for single sphere per frame
        radius=0.04,  # Larger radius for better visibility
        color=(1.0, 0.0, 0.0, 1.0),  # Red with full opacity
        name="Center of Mass"
    )
    
    # Create Center of Pressure visualization (single moving sphere)
    cop_positions_np = cop_total.cpu().numpy()
    # Handle NaN values by setting them to the previous valid position
    cop_display_positions = cop_positions_np.copy()
    for i in range(len(cop_display_positions)):
        if np.isnan(cop_display_positions[i]).any():
            if i > 0:
                cop_display_positions[i] = cop_display_positions[i-1]
            else:
                cop_display_positions[i] = [0, 0, 0]
    
    cop_spheres = Spheres(
        positions=cop_display_positions.reshape(-1, 1, 3),  # (F, 1, 3) for single sphere per frame
        radius=0.03,
        color=(1.0, 0.5, 0.0, 0.9),  # Orange
        name="Center of Pressure"
    )
    
    # Create total GRF arrows at CoP
    print("Creating total GRF arrows...")
    total_force_np = total_force.cpu().numpy()
    grf_origins, grf_tips = create_dynamics_arrows(cop_display_positions, total_force_np, args.force_scale)
    
    total_grf_arrows = Arrows(
        origins=grf_origins,
        tips=grf_tips,
        r_base=0.015,  # Increased size
        r_head=0.025,  # Increased size
        color=(0.0, 0.5, 1.0, 1.0),  # Solid blue
        name="Total GRF at CoP"
    )
    
    # Create required force arrows at CoM (larger and more visible)
    print("Creating required force arrows...")
    req_origins, req_tips = create_dynamics_arrows(com_position, required_force, args.force_scale)
    
    required_force_arrows = Arrows(
        origins=req_origins,
        tips=req_tips,
        r_base=0.015,  # Increased size
        r_head=0.025,  # Increased size
        color=(0.0, 1.0, 0.0, 1.0),  # Solid green
        name="Required Force at CoM"
    )
    
    # Set up viewer
    v = Viewer()
    v.playback_fps = 20
    
    # Add all renderables to the scene
    v.scene.add(skel_seq)
    v.scene.add(right_grf_arrows)
    v.scene.add(left_grf_arrows)
    v.scene.add(com_spheres)
    v.scene.add(cop_spheres)
    v.scene.add(total_grf_arrows)
    v.scene.add(required_force_arrows)
    
    v.run_animations = True
    if cam_pose is not None:
        v.scene.camera.position = cam_pose
    
    print(f"Added visualization with force scale factor: {args.force_scale}")
    
    print("\nVisualization Elements:")
    print("- Black arrows: Individual foot total GRF at CoP") 
    print("- Red sphere: Center of Mass (CoM)")
    print("- Orange sphere: Center of Pressure (CoP)")
    print("- Blue arrows: Total GRF at CoP")
    print("- Green arrows: Required force at CoM")
    print("\nControls:")
    print("- Use mouse to rotate view, scroll to zoom")
    print("- Press SPACE to play/pause animation")
    print("- Press R to reset camera")
    
    v.run() 