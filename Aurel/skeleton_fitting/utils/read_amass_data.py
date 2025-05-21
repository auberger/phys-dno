import numpy as np
import os
import torch
import pickle
from scipy import sparse

# Load SMPL model
def load_smpl_model(model_path):
    """Load the SMPL model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    return model

def batch_rodrigues(rot_vecs):
    """Convert axis-angle representation to rotation matrices.
    Args:
        rot_vecs: Axis-angle representation (batch_size, 3)
    Returns:
        Rotation matrices (batch_size, 3, 3)
    """
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs, dim=1, keepdim=True)
    mask = angle.squeeze(-1) > 1e-8
    
    # For small angles, use Taylor expansion
    if torch.any(~mask):
        # Create identity matrix for small angles
        zeros = torch.zeros_like(rot_vecs)
        I = torch.eye(3, device=rot_vecs.device).unsqueeze(0).expand(batch_size, -1, -1)
        rot_mats = I.clone()
        
        # Apply Taylor expansion for small angles
        if torch.any(mask):
            # Handle non-zero angles
            axis = rot_vecs[mask] / angle[mask]
            
            # Get sin and cos values
            ca = torch.cos(angle[mask])
            sa = torch.sin(angle[mask])
            
            # Compute rotation matrices using Rodrigues formula
            C = 1.0 - ca
            x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
            
            xs, ys, zs = x * sa, y * sa, z * sa
            xC, yC, zC = x * C, y * C, z * C
            xyC, yzC, zxC = x * yC, y * zC, z * xC
            
            valid_rot = torch.zeros((mask.sum(), 3, 3), device=rot_vecs.device)
            
            valid_rot[:, 0, 0] = torch.squeeze(x * xC + ca)
            valid_rot[:, 0, 1] = torch.squeeze(xyC - zs)
            valid_rot[:, 0, 2] = torch.squeeze(zxC + ys)
            valid_rot[:, 1, 0] = torch.squeeze(xyC + zs)
            valid_rot[:, 1, 1] = torch.squeeze(y * yC + ca)
            valid_rot[:, 1, 2] = torch.squeeze(yzC - xs)
            valid_rot[:, 2, 0] = torch.squeeze(zxC - ys)
            valid_rot[:, 2, 1] = torch.squeeze(yzC + xs)
            valid_rot[:, 2, 2] = torch.squeeze(z * zC + ca)
            
            rot_mats[mask] = valid_rot
        
        return rot_mats
    else:
        # All angles are non-zero, proceed with normal calculation
        axis = rot_vecs / (angle + 1e-8)
        
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        
        C = 1.0 - ca
        
        x = axis[..., 0].unsqueeze(-1)
        y = axis[..., 1].unsqueeze(-1)
        z = axis[..., 2].unsqueeze(-1)
        
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        
        rot = torch.zeros((batch_size, 3, 3), device=rot_vecs.device)
        
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        
        return rot

def convert_coordinate_system(joints):
    """
    Convert from SMPL coordinate system to desired coordinate system
    SMPL uses Y-up, we want Z-up with:
    new_y = old_z
    new_z = -old_y
    
    Args:
        joints: Joint positions [frames, joints, 3]
        
    Returns:
        Transformed joint positions [frames, joints, 3]
    """
    # Make a copy to avoid modifying the original
    transformed_joints = joints.copy()
    
    # Store original y
    original_y = transformed_joints[..., 1].copy()
    
    # y = z
    transformed_joints[..., 1] = transformed_joints[..., 2]
    
    # z = -y
    transformed_joints[..., 2] = -original_y
    
    return transformed_joints

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights):
    """Linear Blend Skinning
    Args:
        betas: Shape parameters (batch_size, 10)
        pose: Pose parameters in axis-angle format (batch_size, 24*3)
        v_template: Template mesh vertices (6890, 3)
        shapedirs: Shape blend shapes (6890, 3, 10)
        posedirs: Pose blend shapes (6890, 3, 207)
        J_regressor: Joint regressor (24, 6890)
        parents: Kinematic tree (24)
        lbs_weights: Linear blend skinning weights (6890, 24)
        
    Returns:
        vertices: Mesh vertices (batch_size, 6890, 3)
        joints: Model joints (batch_size, 24, 3)
    """
    batch_size = betas.shape[0]
    device = betas.device
    
    # Add shape contribution
    v_shaped = v_template.unsqueeze(0) + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    
    # Get rest joints from the shaped mesh
    J = torch.matmul(J_regressor, v_shaped.reshape(batch_size, -1, 3))
    
    # Get the rotation matrices for each joint
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, 24, 3, 3)
    
    # Add pose blend shapes
    # Subtract rest pose (first joint is root rotation)
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=device)).reshape(batch_size, -1)
    v_posed = v_shaped + torch.einsum('bl,mkl->bmk', [pose_feature, posedirs])
    
    # Rotation matrices (global)
    J_transformed = torch.zeros((batch_size, 24, 3), device=device)
    J_transformed[:, 0] = J[:, 0]
    
    # Create the transformation matrix for each joint
    transform_mats = torch.zeros((batch_size, 24, 4, 4), device=device)
    transform_mats[:, :, 3, 3] = 1
    
    # The root has no parent, so the global transform is just its local rotation
    transform_mats[:, 0, :3, :3] = rot_mats[:, 0]
    transform_mats[:, 0, :3, 3] = J[:, 0]
    
    # For all other joints, compute the global transform recursively
    for i in range(1, 24):
        parent = parents[i]
        
        # Compute relative location
        loc = J[:, i] - J[:, parent]
        
        # Set local transform
        transform_mats[:, i, :3, :3] = rot_mats[:, i]
        transform_mats[:, i, :3, 3] = loc
        
        # Apply parent transform
        transform_mats[:, i] = torch.matmul(transform_mats[:, parent], transform_mats[:, i])
        
        # Apply to joint
        J_transformed[:, i] = transform_mats[:, i, :3, 3]
    
    # Apply skinning
    T = torch.zeros((batch_size, 6890, 4, 4), device=device)
    for i in range(24):
        T += torch.matmul(lbs_weights[:, i].unsqueeze(-1).unsqueeze(-1), 
                         transform_mats[:, i].reshape(batch_size, 1, 4, 4))
    
    homogen_coord = torch.ones((batch_size, 6890, 1), device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    
    vertices = v_homo[:, :, :3, 0]
    
    return vertices, J_transformed

def process_amass_data(amass_file, smpl_model_path, output_file, frame_range=None, downsample_factor=1):
    """
    Process AMASS data using SMPL model and save joint positions
    
    Args:
        amass_file: Path to the AMASS .npz file
        smpl_model_path: Path to the SMPL model
        output_file: Path to save the output joint positions
        frame_range: Optional tuple (start_frame, end_frame) to process only a subset of frames
        downsample_factor: Optional integer to downsample frames (keep every Nth frame)
    """
    print(f"Processing {amass_file}...")
    
    # Load AMASS data
    data = np.load(amass_file)
    print(f"Keys in the file: {list(data.keys())}")
    
    # Load poses from the file
    poses = data['poses']  # Should be of shape [num_frames, num_joints*3]
    print(f"Loaded {poses.shape[0]} frames of motion data")
    print(f"Pose shape: {poses.shape}")
    
    # Load global translation data for the root/pelvis
    if 'trans' in data:
        trans = data['trans']  # Should be of shape [num_frames, 3]
        print(f"Trans shape: {trans.shape}")
    else:
        print("No translation data found, using zeros")
        trans = np.zeros((poses.shape[0], 3))
    
    # Apply frame range if specified
    if frame_range is not None:
        start_frame, end_frame = frame_range
        
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
        if end_frame > poses.shape[0] or end_frame == -1:
            end_frame = poses.shape[0]
            
        # Extract the specified frames
        poses = poses[start_frame:end_frame]
        trans = trans[start_frame:end_frame]
        print(f"Selected frames {start_frame} to {end_frame} (total: {poses.shape[0]} frames)")
    
    # Apply downsampling if factor > 1
    if downsample_factor > 1:
        # Keep every Nth frame
        poses = poses[::downsample_factor]
        trans = trans[::downsample_factor]
        print(f"Downsampled by factor of {downsample_factor}, new frame count: {poses.shape[0]}")
    
    # AMASS uses more joint parameters than SMPL (156 vs 72)
    # We need to extract just the SMPL pose parameters (first 72)
    if poses.shape[1] > 72:
        print(f"AMASS uses {poses.shape[1]} parameters, extracting only first 72 (SMPL)")
        poses = poses[:, :72]  # Keep only SMPL poses
    
    num_frames = poses.shape[0]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SMPL model
    smpl_model = load_smpl_model(smpl_model_path)
    print("SMPL model loaded successfully")
    
    # Convert data to torch tensors
    poses_tensor = torch.FloatTensor(poses).to(device)
    trans_tensor = torch.FloatTensor(trans).to(device)
    
    # Create zero beta tensor (neutral body shape)
    betas_tensor = torch.zeros((num_frames, 10), device=device)
    
    # Use simpler approach - utilize SMPLH model directly
    from smplx import SMPL
    
    try:
        # Try to use SMPLX library if available
        print("Attempting to use SMPLX library...")
        model = SMPL(model_path=smpl_model_path, gender='neutral')
        
        # Process in batches
        batch_size = 100
        num_batches = (num_frames + batch_size - 1) // batch_size
        all_joints = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_frames)
            
            # Get batch data
            batch_poses = poses_tensor[start_idx:end_idx]
            batch_betas = betas_tensor[start_idx:end_idx]
            batch_trans = trans_tensor[start_idx:end_idx]
            
            # Forward pass
            output = model(global_orient=batch_poses[:, :3], 
                          body_pose=batch_poses[:, 3:], 
                          betas=batch_betas,
                          transl=batch_trans)  # Include global translation
            
            # Store joints - get only the original 24 SMPL joints
            # SMPLX gives 45 joints, but we only want the first 24 which are the original SMPL joints
            batch_joints = output.joints[:, :24, :].detach().cpu().numpy()
            all_joints.append(batch_joints)
            
            print(f"Processed batch {batch_idx+1}/{num_batches}")
        
        # Concatenate all batches
        joints = np.concatenate(all_joints, axis=0)
        
    except ImportError:
        print("SMPLX library not available. Using manual implementation...")
        # Extract SMPL model components for manual implementation
        v_template = torch.FloatTensor(smpl_model['v_template']).to(device)
        
        # Handle shapedirs which might be in different formats
        if isinstance(smpl_model['shapedirs'], np.ndarray):
            shapedirs = torch.FloatTensor(smpl_model['shapedirs']).to(device)
        else:
            print("Converting shapedirs from sparse matrix...")
            shapedirs = torch.FloatTensor(smpl_model['shapedirs'].todense()).to(device)
        
        if 'posedirs' in smpl_model:
            if sparse.issparse(smpl_model['posedirs']):
                posedirs = torch.FloatTensor(smpl_model['posedirs'].todense()).to(device)
            else:
                posedirs = torch.FloatTensor(smpl_model['posedirs']).to(device)
        else:
            print("Warning: posedirs not found in SMPL model. Using zeros.")
            posedirs = torch.zeros((v_template.shape[0], 3, 207), device=device)
        
        # J_regressor might be sparse
        if sparse.issparse(smpl_model['J_regressor']):
            J_regressor = torch.FloatTensor(smpl_model['J_regressor'].todense()).to(device)
        else:
            J_regressor = torch.FloatTensor(smpl_model['J_regressor']).to(device)
        
        parents = smpl_model['kintree_table'][0].astype(np.int64)
        parents = torch.LongTensor(parents).to(device)
        
        lbs_weights = torch.FloatTensor(smpl_model['weights']).to(device)
        
        # We'll use a simplified forward pass
        print("Computing joint positions manually...")
        
        # Add shape contribution
        v_shaped = v_template.unsqueeze(0).expand(num_frames, -1, -1) + torch.einsum('bi,ijk->bjk', [betas_tensor, shapedirs])
        
        # Get rest joints from the shaped mesh
        J = torch.matmul(J_regressor, v_shaped.reshape(num_frames, -1, 3))
        
        # Reshape poses for rodrigues
        pose_rots = batch_rodrigues(poses_tensor.reshape(-1, 3)).reshape(num_frames, 24, 3, 3)
        
        # Create global rotation matrices
        J_transformed = torch.zeros((num_frames, 24, 3), device=device)
        rot_mats = torch.zeros((num_frames, 24, 4, 4), device=device)
        rot_mats[:, :, 3, 3] = 1.0
        
        # Set root transformation
        rot_mats[:, 0, :3, :3] = pose_rots[:, 0]
        J_transformed[:, 0] = J[:, 0]
        rot_mats[:, 0, :3, 3] = J[:, 0]
        
        # Forward kinematics
        for i in range(1, 24):
            parent = parents[i]
            
            # Get joint location in rest pose
            rot_mats[:, i, :3, :3] = pose_rots[:, i]
            rot_mats[:, i, :3, 3] = J[:, i] - J[:, parent]
            
            # Apply parent rotation
            rot_mats[:, i] = torch.matmul(rot_mats[:, parent], rot_mats[:, i])
            
            # Translation after rotation
            J_transformed[:, i] = rot_mats[:, i, :3, 3]
        
        # Add global translation to all joints
        J_transformed = J_transformed + trans_tensor.unsqueeze(1)
        
        # Store joints
        joints = J_transformed.detach().cpu().numpy()
    
    print(f"Joint shape before coordinate conversion: {joints.shape}")
    
    # Convert coordinate system: y=z, z=-y
    joints = convert_coordinate_system(joints)
    
    print(f"Final joint shape after coordinate conversion: {joints.shape}")
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, joints)
    print(f"Saved joint positions to {output_file}")
    
    return joints

if __name__ == "__main__":
    # File paths
    amass_file = 'dataset/CMU/13/13_29_poses.npz'
    smpl_model_path = 'body_models/smpl/SMPL_NEUTRAL.pkl'
    
    # Example: Process specific frames with downsampling
    frame_range = (3800, 4160)
    downsample_factor = 3  # Keep every 2nd frame
    output_file = 'Aurel/skeleton_fitting/output/amass_joints/CMU_13_29_frames_3800_4160_ds3.npy'
    
    # Process the data
    joints = process_amass_data(amass_file, smpl_model_path, output_file, frame_range, downsample_factor)
    
    # Print shape of the resulting joints
    print(f"Output joints shape: {joints.shape}")
    