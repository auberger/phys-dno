import numpy as np
import torch
import os
import sys

# Paths
FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(FILE_DIR, "../../..")
sys.path.append(ROOT_DIR)

from model.rotation2xyz import Rotation2xyz
from visualize.simplify_loc2rot import joints2smpl

def convert_joints_to_vertices(npy_path, sample_idx=0, rep_idx=0, device_id=0, cuda=True):
    """
    Convert 3D joint positions from a results.npy file to SMPL vertices
    
    Args:
        npy_path (str): Path to the results.npy file
        sample_idx (int): Sample index to process
        rep_idx (int): Repetition index to process
        device_id (int): GPU device ID
        cuda (bool): Whether to use CUDA
        
    Returns:
        dict: Dictionary containing SMPL vertices and other data
    """
    # Load the motion data
    motions = np.load(npy_path, allow_pickle=True)
    if npy_path.endswith('.npz'):
        motions = motions['arr_0']
    motions = motions[None][0]
    
    # Initialize the rotation to xyz converter
    rot2xyz = Rotation2xyz(device='cpu')
    faces = rot2xyz.smpl_model.faces
    
    # Get the motion dimensions
    bs, njoints, nfeats, nframes = motions['motion'].shape
    
    # Calculate the absolute index
    total_num_samples = motions['num_samples']
    absl_idx = sample_idx * total_num_samples + rep_idx
    
    # Get the actual number of frames for this motion
    num_frames = motions['motion'][absl_idx].shape[-1]
    real_num_frames = motions['lengths'][absl_idx]
    
    # Process the motion data based on the feature dimension
    if nfeats == 3:
        print(f'Running faster SMPLify (beta fixed, reduced iterations) for sample [{sample_idx}], repetition [{rep_idx}].')
        
        # Initialize the joints to SMPL converter
        j2s = joints2smpl(num_frames=num_frames, device_id=device_id, cuda=cuda)
        
        # Convert joint positions to SMPL parameters
        motion_tensor, opt_dict = j2s.joint2smpl(motions['motion'][absl_idx].transpose(2, 0, 1))
        processed_motion = motion_tensor.cpu().numpy()
    elif nfeats == 6:
        # If already in rotation format, just use it directly
        processed_motion = motions['motion'][[absl_idx]]
    else:
        raise ValueError(f"Unexpected feature dimension: {nfeats}. Expected 3 (joints) or 6 (rotations).")
    
    # Convert SMPL parameters to vertices
    vertices = rot2xyz(torch.tensor(processed_motion), mask=None,
                       pose_rep='rot6d', translation=True, glob=True,
                       jointstype='vertices', vertstrans=True)
    
    # Create result dictionary
    result = {
        'vertices': vertices[0, :, :, :real_num_frames].cpu().numpy(),  # [num_vertices, 3, num_frames]
        'faces': faces,
        'motion_params': processed_motion[0, :, :, :real_num_frames],  # SMPL parameters
        'text': motions['text'][sample_idx],
        'length': real_num_frames
    }
    
    return result

# Example usage (can be commented out when imported as a module)
if __name__ == "__main__":
    # Example file path
    npy_path = "Aurel/skeleton_fitting/dno_example_output_save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/results.npy"
    
    # Convert joints to vertices
    result = convert_joints_to_vertices(npy_path, sample_idx=0, rep_idx=0, cuda=False)
    
    # Access the vertices
    vertices = result['vertices']  # Shape: [num_vertices, 3, num_frames]
    
    # Print some information
    print(f"Converted {result['length']} frames of motion")
    print(f"Vertex shape: {vertices.shape}")
    print(f"Associated text: {result['text']}")
    
    # Optionally save the result
    np.save("vertices_output.npy", result)