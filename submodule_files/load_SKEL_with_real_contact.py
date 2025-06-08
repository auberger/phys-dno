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

def create_grf_arrows(sphere_positions: torch.Tensor, forces: torch.Tensor, 
                     force_scale: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """
    Create GRF arrows from sphere forces.
    
    Args:
        sphere_positions: (F, N, 3) tensor of sphere positions
        forces: (F, N, 3) tensor of per-sphere forces
        force_scale: Scale factor to make forces visible
    
    Returns:
        Tuple of (arrow_origins, arrow_tips) as numpy arrays
    """
    F, num_spheres, _ = sphere_positions.shape
    
    # Convert to numpy
    positions = sphere_positions.cpu().numpy()
    forces_np = forces.cpu().numpy()
    
    # Only create arrows where there's significant force
    force_magnitudes = np.linalg.norm(forces_np, axis=2)
    significant_force_mask = force_magnitudes > 1.0  # Minimum 1N
    
    # Count significant forces per frame
    arrows_per_frame = np.sum(significant_force_mask, axis=1)
    max_arrows = np.max(arrows_per_frame)
    
    # Initialize arrow arrays
    arrow_origins = np.zeros((F, max_arrows, 3))
    arrow_tips = np.zeros((F, max_arrows, 3))
    
    for frame in range(F):
        valid_spheres = np.where(significant_force_mask[frame])[0]
        
        for i, sphere_idx in enumerate(valid_spheres):
            if i < max_arrows:
                origin = positions[frame, sphere_idx]
                force_vec = forces_np[frame, sphere_idx] * force_scale
                
                arrow_origins[frame, i] = origin
                arrow_tips[frame, i] = origin + force_vec
    
    return arrow_origins, arrow_tips

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a SKEL model with real contact spheres and GRF arrows.")
    parser.add_argument("-s", "--motion_file", type=str, help="Path to a skel motion file", required=True)
    parser.add_argument("-c", "--contact_file", type=str, help="Path to contact forces file", required=True)
    parser.add_argument("-z", "--z_up", help="Rotate the mesh 90 deg", action="store_true")
    parser.add_argument("--force_scale", type=float, default=0.001, 
                       help="Scale factor for GRF arrows (default: 0.001)")
    parser.add_argument("--sphere_radius", type=float, default=0.015, 
                       help="Radius of contact spheres in meters (default: 0.015)")

    args = parser.parse_args()

    assert os.path.exists(args.motion_file), f"Could not find {args.motion_file}"
    assert os.path.exists(args.contact_file), f"Could not find {args.contact_file}"
    assert args.motion_file.endswith(".pkl"), "Motion file should be a .pkl file"
    
    # Load SKEL sequence
    skel_seq = SKELSequence.from_file(args.motion_file, name="SKEL", fps_in=120, fps_out=120, z_up=args.z_up)
    cam_pose = np.array([0, 1.2, 4.0])
    
    # Load contact data
    print("Loading contact data...")
    contact_data = torch.load(args.contact_file)
    sphere_positions = contact_data['sphere_positions']
    sphere_forces = contact_data['sphere_forces']  # Use sphere-specific forces
    
    # Split spheres into right and left (assuming first half is right foot)
    n_spheres = sphere_positions.shape[1]
    mid_idx = n_spheres // 2
    
    # Create contact spheres visualization
    # Right foot spheres (grey)
    right_sphere_positions = sphere_positions[:, :mid_idx, :].cpu().numpy()
    right_spheres = Spheres(
        positions=right_sphere_positions,
        radius=args.sphere_radius,
        color=(0.2, 0.2, 0.2, 0.8),  # Dark grey with transparency
        name="Right Contact Spheres"
    )
    
    # Left foot spheres (grey)
    left_sphere_positions = sphere_positions[:, mid_idx:, :].cpu().numpy()
    left_spheres = Spheres(
        positions=left_sphere_positions,
        radius=args.sphere_radius,
        color=(0.2, 0.2, 0.2, 0.8),  # Dark grey with transparency
        name="Left Contact Spheres"
    )
    
    # Create GRF arrows
    print("Creating GRF arrows...")
    arrow_origins, arrow_tips = create_grf_arrows(sphere_positions, sphere_forces, args.force_scale)  # Use sphere_forces
    
    # Filter out zero arrows (where no force was applied)
    non_zero_arrows_mask = np.linalg.norm(arrow_tips - arrow_origins, axis=2) > 1e-6
    
    # Create arrows only where there are actual forces
    grf_arrows = None
    if np.any(non_zero_arrows_mask):
        # Create arrows with red color for forces
        grf_arrows = Arrows(
            origins=arrow_origins,
            tips=arrow_tips,
            r_base=0.005,  # Thin base
            r_head=0.01,   # Slightly thicker head
            color=(1.0, 0.0, 0.0, 0.9),  # Red arrows
            name="Ground Reaction Forces"
        )
    
    # Set up viewer
    v = Viewer()
    v.playback_fps = 20
    v.scene.add(skel_seq)
    v.scene.add(right_spheres)
    v.scene.add(left_spheres)
    
    if grf_arrows is not None:
        v.scene.add(grf_arrows)
        print(f"Added GRF arrows with scale factor: {args.force_scale}")
    else:
        print("No GRF arrows to display (no significant forces found)")
    
    v.run_animations = True
    if cam_pose is not None:
        v.scene.camera.position = cam_pose
    
    print("\nVisualization Controls:")
    print("- Blue spheres: Right foot contact points")
    print("- Green spheres: Left foot contact points") 
    print("- Red arrows: Ground Reaction Forces")
    print("- Use mouse to rotate view, scroll to zoom")
    print("- Press SPACE to play/pause animation")
    print("- Press R to reset camera")
    
    v.run() 