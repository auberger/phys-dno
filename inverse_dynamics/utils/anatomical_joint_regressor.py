import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from scipy import sparse
from sklearn.linear_model import Lasso, LassoCV
from skel.skel_model import SKEL
import skel.kin_skel as kin_skel

class SparseSMPLtoAnatomicalRegressor:
    def __init__(self, output_dir="output/regressor", gender="male", debug=False):
        self.debug = debug
        self.output_dir = output_dir
        self.gender = gender
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (self.debug is True):
            print(f"Using device: {self.device}")
            print(f"Using gender: {self.gender}")
        
        # Load models
        self.load_models()

    def load_models(self):
        """Load SKEL model"""
        # SKEL model for both SMPL and OpenSim joint regressors
        self.skel_model = SKEL(gender=self.gender).to(self.device)

        # Get joint regressors
        self.j_regressor_osim = self.skel_model.J_regressor_osim
        self.j_regressor_smpl = self.skel_model.J_regressor.to_dense() # Convert sparse J_regressor to dense for use with einsum

        if (self.debug is True):
            print("SKEL model loaded successfully")
            print(f"J_regressor_smpl type: {type(self.j_regressor_smpl)}")
            print(f"J_regressor_smpl shape: {self.j_regressor_smpl.shape}")
            print(f"J_regressor_osim type: {type(self.j_regressor_osim)}")
            print(f"J_regressor_osim shape: {self.j_regressor_osim.shape}")

    def forward_kinematics(self, pose_axis_angle, shape, trans=None):
        """
        SKEL forward kinematics to generate vertices and joints
        
        Args:
            pose_axis_angle: Pose parameters in axis-angle format (batch_size, 46)
            shape: Shape parameters (batch_size, 10)
            trans: Translation parameters (batch_size, 3)
            
        Returns:
            vertices: Mesh vertices (batch_size, 6890, 3)
            smpl_joints: SMPL joints (batch_size, 24, 3)
            osim_joints: OpenSim joints (batch_size, 24, 3)
        """
        batch_size = pose_axis_angle.shape[0]
        
        # SKEL only accepts 'skel' or 'bsm' pose types, not 'smpl'
        output = self.skel_model.forward(
            poses=pose_axis_angle,
            betas=shape,
            trans=trans if trans is not None else torch.zeros((batch_size, 3), device=self.device),
            poses_type='skel',  # Use SKEL pose format
            skelmesh=True       # Get skin mesh
        )
        
        # Get vertices from the output
        verts = output.skin_verts
        
        # Compute SMPL joints using the joint regressor
        smpl_joints = torch.einsum("bik,ji->bjk", [verts, self.j_regressor_smpl])
        
        # OpenSim joints are already computed by SKEL
        osim_joints = output.joints
        #osim_joints = torch.einsum("bik,ji->bjk", [verts, self.j_regressor_osim])

        return verts, smpl_joints, osim_joints

    def generate_data(self, num_samples=10000):
        """Generate paired SMPL and OpenSim joint data"""
        if (self.debug is True): print(f"Generating {num_samples} samples...")
        
        # Arrays to store data
        smpl_joints_all = []
        osim_joints_all = []
        
        # Get the number of pose parameters for SKEL
        num_pose_params = self.skel_model.num_q_params
        if (self.debug is True): print(f"SKEL uses {num_pose_params} pose parameters")
        
        # SKEL pose parameter names and limits
        pose_param_names = kin_skel.pose_param_names
        
        # Define pose limits - for parameters not explicitly defined, use default range
        pose_limits = kin_skel.pose_limits
        
        # Generate random poses with a smart sampling strategy
        # Define pose categories to ensure diversity
        pose_categories = {
            'neutral': 0.4,       # Neutral standing poses (40%)
            'bent': 0.15,         # Bent poses with significant hip/knee flexion (15%)
            'asymmetric': 0.15,   # Asymmetric poses with different left/right positions (15%)
            'arm_varied': 0.15,   # Poses with varied arm positions (15%)
            'extreme': 0.15       # Poses that explore extremes of joint ranges (15%)
        }
        
        for i in tqdm(range(num_samples)):
            # Sample a pose category based on distribution
            category = np.random.choice(
                list(pose_categories.keys()),
                p=list(pose_categories.values())
            )
            
            # Shape parameters (betas)
            # 90% zero beta, 10% small variations
            if np.random.rand() < 0.9:
                beta = torch.zeros(1, 10).to(self.device)
            else:
                # For body shape, first component controls height, second controls weight
                beta = torch.FloatTensor(1, 10).uniform_(-0.5, 0.5).to(self.device)
                # Occasionally generate more extreme body shapes
                if np.random.rand() < 0.2:
                    beta[0, 0] = torch.FloatTensor(1).uniform_(-1.5, 1.5).to(self.device)  # Height
                    beta[0, 1] = torch.FloatTensor(1).uniform_(-1.5, 1.5).to(self.device)  # Weight
            
            # Create pose parameters according to anatomical constraints
            pose = torch.zeros(1, num_pose_params).to(self.device)
            
            # Base sampling ranges depend on pose category
            if category == 'neutral':
                # More concentration around zero
                range_factor = 0.3
            elif category == 'bent':
                # Focus on hip and knee bending
                range_factor = 0.5
                # Additional hip and knee flexion
                hip_knee_factor = 0.7
            elif category == 'asymmetric':
                # Different ranges for left vs right
                range_factor = 0.6
                # Will use different factors for left vs right
            elif category == 'arm_varied':
                # Focus on arm variation
                range_factor = 0.4
                # Higher range for arm joints
                arm_factor = 0.8
            else:  # 'extreme'
                # Use full ranges
                range_factor = 1.0
            
            # Apply different sampling strategies based on pose category
            for param_idx, param_name in enumerate(pose_param_names):

                if param_name in pose_limits:
                    min_val, max_val = pose_limits[param_name]
                    
                    # Modify range based on category and joint
                    param_range_factor = range_factor
                    
                    # Category-specific modifications
                    if category == 'bent' and ('hip_flexion' in param_name or 'knee_angle' in param_name):
                        param_range_factor = hip_knee_factor
                        # Bias toward flexion for bent poses
                        if 'hip_flexion' in param_name or 'knee_angle' in param_name:
                            min_val = max(0, min_val)  # Ensure positive flexion
                            
                    elif category == 'asymmetric':
                        # Make left and right sides different
                        if '_r' in param_name:
                            param_range_factor = 0.7
                        elif '_l' in param_name:
                            param_range_factor = 0.5
                            
                    elif category == 'arm_varied' and ('shoulder' in param_name or 'elbow' in param_name or 'wrist' in param_name):
                        param_range_factor = arm_factor
                        
                    elif category == 'extreme':
                        # For extreme poses, occasionally use values very close to limits
                        if np.random.rand() < 0.3:  # 30% chance
                            # Pick either min or max to get close to
                            if np.random.rand() < 0.5:
                                # Close to min
                                min_val_adj = min_val
                                max_val_adj = min_val + (max_val - min_val) * 0.2
                            else:
                                # Close to max
                                min_val_adj = max_val - (max_val - min_val) * 0.2
                                max_val_adj = max_val
                                
                            pose[0, param_idx] = torch.FloatTensor(1).uniform_(min_val_adj, max_val_adj).to(self.device)
                            continue  # Skip the normal range adjustment
                    
                    # Apply the range factor
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) / 2
                    min_val_adj = center - half_range * param_range_factor
                    max_val_adj = center + half_range * param_range_factor
                    
                    # Generate value from range
                    pose[0, param_idx] = torch.FloatTensor(1).uniform_(min_val_adj, max_val_adj).to(self.device)
                else:
                    # Default range for unspecified parameters
                    pose[0, param_idx] = torch.FloatTensor(1).uniform_(-0.5, 0.5).to(self.device)
            
            # Save the first pose for debugging
            if i == 0:
                np.save(os.path.join(self.output_dir, f"test_pose_{self.gender}.npy"), pose.detach().cpu().numpy())
                print(f"Saved test pose with shape {pose.shape}")
            
            # Random translation for the pelvis
            trans = torch.FloatTensor(1, 3).uniform_(-1.0, 1.0).to(self.device)

            # Forward kinematics with translation
            _, smpl_joints, osim_joints = self.forward_kinematics(pose, beta, trans)
            
            # Save
            smpl_joints_all.append(smpl_joints.detach().cpu().numpy())
            osim_joints_all.append(osim_joints.detach().cpu().numpy())
            
        # Convert to arrays
        smpl_joints_all = np.vstack(smpl_joints_all)
        osim_joints_all = np.vstack(osim_joints_all)
        
        print(f"Generated data shapes: SMPL {smpl_joints_all.shape}, OpenSim {osim_joints_all.shape}")
        
        # Save raw data
        np.save(os.path.join(self.output_dir, f"smpl_joints_data_{self.gender}.npy"), smpl_joints_all)
        np.save(os.path.join(self.output_dir, f"osim_joints_data_{self.gender}.npy"), osim_joints_all)

        return smpl_joints_all, osim_joints_all

    def train_sparse_regressor(self, smpl_joints, osim_joints, alpha=0.005):
        """Train a sparse regressor from SMPL to OpenSim joints"""
        print("Training sparse regressor...")
        
        # 3. Normalize joint positions for uniform scaling
        smpl_joints_mean = np.mean(smpl_joints, axis=0)
        smpl_joints_std = np.std(smpl_joints, axis=0)
        smpl_joints_norm = (smpl_joints - smpl_joints_mean) / (smpl_joints_std + 1e-8)
        
        osim_joints_mean = np.mean(osim_joints, axis=0)
        osim_joints_std = np.std(osim_joints, axis=0)
        osim_joints_norm = (osim_joints - osim_joints_mean) / (osim_joints_std + 1e-8)
        
        print("Normalized joint positions for training")
        
        # Reshape data for regression
        # Each anatomical joint (24x3) as function of all SMPL joints (24x3)
        X = smpl_joints_norm.reshape(-1, 24*3)  # Each sample has all SMPL joints flattened
        
        # Define joint hierarchy for very strict sparsity
        # For each joint, define the corresponding SMPL joint and parent joints
        # Format: [joint_idx, [list of valid SMPL joints that can influence it]]
        joint_hierarchy = {
            # Pelvis
            0: [0, 1, 2, 3],  # Pelvis depends on itself and its children
            
            # Right leg
            1: [0, 1, 2, 5],  # Right hip: SMPL pelvis, left hip, right hip, right knee
            2: [0, 2, 5, 8],  # Right knee: SMPL pelvis, right knee, right hip, right tibia
            3: [0, 2, 5, 8, 11],  # Right ankle: SMPL pelvis, right ankle, right knee, right hip, right foot
            4: [0, 2, 5, 8, 11],     # Right calcn: SMPL pelvis, right ankle, right knee, right hip, right foot
            5: [0, 2, 5, 8, 11],    # Right toes: SMPL pelvis, right ankle, right knee, right hip, right foot
            
            # Left leg
            6: [0, 1, 2, 4],  # Left hip: SMPL pelvis, left hip, right hip, left knee
            7: [0, 1, 4, 7],  # Left knee: SMPL pelvis, left knee, left hip, left tibia
            8: [0, 1, 4, 7, 10],  # Left ankle: SMPL pelvis, left ankle, left knee, left hip, left foot
            9: [0, 1, 4, 7, 10],     # Left calcn: SMPL pelvis, left ankle, left knee, left hip
            10: [0, 1, 4, 7, 10],   # Left toes: SMPL pelvis, left ankle, left knee, left hip, left foot
            
            # Spine
            11: [0, 3, 6],  # Lumbar: SMPL pelvis, spine1, spine2
            12: [0, 3, 6, 9], # Thorax: SMPL pelvis, spine1, spine2, spine3
            
            # Head and neck
            13: [0, 3, 6, 9, 13, 14, 12, 15],   # Head: SMPL pelvis, spine2, spine3, neck, left collar, right collar
            
            # Right arm and shoulder
            14: [0, 9, 12, 14, 17], # Right scapula: SMPL pelvis, spine3, neck, left collar, right collar, right humerus
            15: [0, 9, 12, 14, 17, 19], # Right humerus: SMPL pelvis, spine3, neck, left collar, right collar, right humerus, right elbow
            16: [0, 9, 12, 14, 17, 19, 21], # Right ulna: SMPL pelvis, spine3, neck, left collar, right collar, right humerus, right elbow, right wrist
            17: [0, 9, 12, 14, 17, 19, 21], # Right radius: SMPL pelvis, spine3, neck, left collar, right collar, right humerus, right elbow, right wrist
            18: [0, 9, 12, 14, 17, 19, 21], # Right hand: SMPL pelvis, spine3, neck, left collar, right collar, right humerus, right elbow, right wrist
            
            # Left arm and shoulder
            19: [0, 9, 12, 13, 16], # Left scapula: SMPL pelvis, spine3, neck, left collar, left humerus
            20: [0, 9, 12, 13, 16, 18], # Left humerus: SMPL pelvis, spine3, neck, left collar, left humerus
            21: [0, 9, 12, 13, 16, 18, 20], # Left ulna: SMPL pelvis, spine3, neck, left collar, left humerus, left elbow
            22: [0, 9, 12, 13, 16, 18, 20], # Left radius: SMPL pelvis, spine3, neck, left collar, left humerus, left elbow
            23: [0, 9, 12, 13, 16, 18, 20], # Left hand: SMPL pelvis, spine3, neck, left collar, left humerus, left elbow, left wrist
        }
        
        print("Created strict joint dependency constraints")
        
        # Create feature groups based on joint hierarchy
        allowed_features = []
        for joint_idx in range(24):
            # Get allowed SMPL joints for this joint
            allowed_smpl_joints = joint_hierarchy[joint_idx]
            
            # Create indices for all features from allowed joints (all coordinates)
            allowed_joint_features = []
            for j in allowed_smpl_joints:
                allowed_joint_features.extend([j*3, j*3+1, j*3+2])
                
            allowed_features.append(allowed_joint_features)
        
        # Train separate regressor for each anatomical joint coordinate
        regressors = []
        scaler_info = {
            'smpl_mean': smpl_joints_mean,
            'smpl_std': smpl_joints_std,
            'osim_mean': osim_joints_mean,
            'osim_std': osim_joints_std
        }
        
        # Train a regressor for each anatomical joint
        for joint_idx in tqdm(range(24)):
            y = osim_joints_norm[:, joint_idx, :].reshape(-1, 3)
            
            # Get the allowed features for this joint
            allowed_joint_features = allowed_features[joint_idx]
            
            # Train a regressor for each coordinate (x, y, z)
            joint_regressors = []
            for coord_idx in range(3):
                # Build a mask matrix where only allowed features have normal weight
                # Create a feature matrix with zeros for all disallowed features
                X_masked = X.copy()
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[allowed_joint_features] = True
                
                # Zero out columns that aren't allowed
                for i in range(X.shape[1]):
                    if not mask[i]:
                        X_masked[:, i] = 0
                
                # Train on the masked data
                model = Lasso(alpha=alpha, max_iter=10000, selection='random')
                model.fit(X_masked, y[:, coord_idx])
                
                # Double-check: zero out any coefficients for disallowed features
                for i in range(X.shape[1]):
                    if not mask[i]:
                        model.coef_[i] = 0.0
                        
                joint_regressors.append(model)

            regressors.append(joint_regressors)
        
        # Save regressor with scaling information
        regressor_data = {
            'regressors': regressors,
            'scalers': scaler_info,
            'joint_hierarchy': joint_hierarchy  # Save the hierarchy for reference
        }
        
        with open(os.path.join(self.output_dir, f"smpl_to_osim_regressor_{self.gender}.pkl"), 'wb') as f:
            pickle.dump(regressor_data, f)
            
        print("Regressor trained and saved successfully")
        
        # Analyze sparsity
        self._analyze_regressor_sparsity(regressors, joint_hierarchy)
        
        return regressors, scaler_info
    
    def _analyze_regressor_sparsity(self, regressors, joint_hierarchy=None):
        """Analyze the sparsity of the regressor"""
        print("\nRegressor sparsity analysis:")
        
        total_coeffs = 24 * 3 * 24 * 3  # 24 joints x 3 coords x (24 joints x 3 coords)
        nonzero_coeffs = 0
        
        # Mapping from joint indices to names for better interpretability (SMPL joint names)
        joint_names = [
            "pelvis", "l_femur", "r_femur", "spine1", 
            "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", 
            "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", 
            "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", 
            "l_wrist", "r_wrist", "l_hand", "r_hand"
        ]
        
        # For each anatomical joint
        for i, joint_regressors in enumerate(regressors):
            # Average number of non-zero coefficients across x, y, z coordinates
            nonzeros = []
            for coord_idx, model in enumerate(joint_regressors):
                coef = model.coef_
                nonzeros.append(np.sum(coef != 0))
                nonzero_coeffs += np.sum(coef != 0)
            
            avg_nonzero = np.mean(nonzeros)
            print(f"Joint {i} ({joint_names[i]}) - avg non-zero coefficients: {avg_nonzero:.1f}/{24*3} ({100*avg_nonzero/(24*3):.1f}%)")
            
            if joint_hierarchy:
                allowed_joints = joint_hierarchy[i]
                print(f"  Allowed SMPL joints: {[joint_names[j] for j in allowed_joints]}")
            
            # List top 3 contributors for each joint
            print("  Top contributors:")
            for coord_idx, model in enumerate(joint_regressors):
                coef = model.coef_
                # Get indices of top 3 contributing SMPL joints for this coordinate
                top_idx = np.argsort(np.abs(coef))[-3:][::-1]
                
                for idx in top_idx:
                    smpl_joint_idx = idx // 3
                    smpl_coord_idx = idx % 3
                    coord_name = ['x', 'y', 'z'][coord_idx]
                    smpl_coord_name = ['x', 'y', 'z'][smpl_coord_idx]
                    print(f"    {coord_name} ← {joint_names[smpl_joint_idx]}.{smpl_coord_name} ({coef[idx]:.3f})")
        
        # Overall sparsity
        sparsity = 100 * (1 - nonzero_coeffs / total_coeffs)
        print(f"\nOverall sparsity: {sparsity:.1f}% ({nonzero_coeffs} non-zero coefficients out of {total_coeffs})")
        
        # Show theoretical maximum sparsity based on joint hierarchy
        if joint_hierarchy:
            allowed_features_count = 0
            for joint_idx in range(24):
                allowed_features_count += len(joint_hierarchy[joint_idx]) * 3 * 3  # Each allowed joint has 3 coords, each output has 3 coords
            
            max_sparsity = 100 * (1 - allowed_features_count / total_coeffs)
            print(f"Maximum possible sparsity with strict constraints: {max_sparsity:.1f}%")
            print(f"Allowed {allowed_features_count} of {total_coeffs} coefficients")
    
    def test_regressor(self, regressors, smpl_joints, scaler_info=None):
        """Test the regressor on some samples"""
        print("\nTesting regressor...")
        
        # Apply normalization
        smpl_joints_norm = (smpl_joints - scaler_info['smpl_mean']) / (scaler_info['smpl_std'] + 1e-8)
        X = smpl_joints_norm.reshape(-1, 24*3)
        
        # Predict using the regressor
        predicted_joints_norm = np.zeros((X.shape[0], 24, 3))
        
        for joint_idx, joint_regressors in enumerate(regressors):
            for coord_idx, model in enumerate(joint_regressors):
                predicted_joints_norm[:, joint_idx, coord_idx] = model.predict(X)
        
        # Denormalize predictions 
        predicted_joints = predicted_joints_norm * scaler_info['osim_std'] + scaler_info['osim_mean']

        
        return predicted_joints
    
    def run_pipeline(self, num_samples=10000, alpha=0.005):
        """Run the full pipeline"""
        # Generate data
        smpl_joints, osim_joints = self.generate_data(num_samples)
        
        # Train regressor with improved settings
        regressors, scaler_info = self.train_sparse_regressor(smpl_joints, osim_joints, alpha)
        
        # Test on a subset
        test_idx = np.random.choice(smpl_joints.shape[0], size=100, replace=False)
        predicted_joints = self.test_regressor(regressors, smpl_joints[test_idx], scaler_info)
        
        # Compute error
        mse = np.mean((predicted_joints - osim_joints[test_idx])**2)
        rmse = np.sqrt(mse)
        print(f"Test RMSE: {rmse:.6f}")
        
        # Save test results
        np.save(os.path.join(self.output_dir, f"test_predictions_{self.gender}.npy"), predicted_joints)
        np.save(os.path.join(self.output_dir, f"test_ground_truth_{self.gender}.npy"), osim_joints[test_idx])
        
        return regressors, scaler_info
    
    @staticmethod
    def predict_anatomical_joints_from_file(smpl_joints_file, regressor_path, gender="male", output_file=None, trial=0):
        """
        Predict anatomical joints from SMPL joints using the trained regressor.
        
        Args:
            smpl_joints_file: Path to the .npy file containing SMPL joint positions [trials, joints, dims, frames]
            regressor_path: Path to directory containing the trained regressor
            gender: Gender used for the regressor ('male' or 'female')
            output_file: Optional path to save the predictions
            trial: Trial index to process (default: 0)
            
        Returns:
            Predicted anatomical joint positions [frames, 24, 3]
        """
        # Load regressor
        regressor_file = os.path.join(regressor_path, f"smpl_to_osim_regressor_{gender}.pkl")
        with open(regressor_file, 'rb') as f:
            regressor_data = pickle.load(f)
        
        regressors = regressor_data['regressors']
        scaler_info = regressor_data['scalers']
        
        # Load SMPL joints with shape (trials, joints, dims, frames)
        try:
            smpl_joints_data = np.load(smpl_joints_file, allow_pickle=True).item()['motion']

            # Select the specified trial and transpose to [frames, joints, dims]
            smpl_joints = smpl_joints_data[trial].transpose(2, 0, 1)
        except:
            smpl_joints = np.load(smpl_joints_file, allow_pickle=True)

        # Ensure we have 24 joints as expected by the regressor (pad if necessary)
        num_frames, num_joints, dims = smpl_joints.shape
        if num_joints < 24:
            # Pad with zeros to reach 24 joints
            padding = np.zeros((num_frames, 24 - num_joints, dims))
            smpl_joints = np.concatenate([smpl_joints, padding], axis=1)
        elif num_joints > 24:
            # Truncate to 24 joints
            smpl_joints = smpl_joints[:, :24, :]
        
        # Normalize input joints
        smpl_joints_norm = (smpl_joints - scaler_info['smpl_mean']) / (scaler_info['smpl_std'] + 1e-8)
        
        # Reshape for regressor [frames, 24*3]
        X = smpl_joints_norm.reshape(num_frames, -1)
        
        # Predict anatomical joints
        pred_joints = np.zeros((num_frames, 24, 3))
        for joint_idx, joint_regressors in enumerate(regressors):
            for coord_idx, model in enumerate(joint_regressors):
                pred_joints[:, joint_idx, coord_idx] = model.predict(X)
        
        # Denormalize predictions
        pred_joints = pred_joints * scaler_info['osim_std'] + scaler_info['osim_mean']
        
        # Save if requested
        if output_file:
            np.save(output_file, pred_joints)
            print(f"Saved predictions to {output_file}")
        
        return pred_joints

    @staticmethod
    def predict_anatomical_joints(smpl_joints, regressor_path, gender="male", output_file=None, trial=0):
        """
        Predict anatomical joints from SMPL joints using the trained regressor.
        
        Args:
            smpl_joints_file: Path to the .npy file containing SMPL joint positions [trials, joints, dims, frames]
            regressor_path: Path to directory containing the trained regressor
            gender: Gender used for the regressor ('male' or 'female')
            output_file: Optional path to save the predictions
            trial: Trial index to process (default: 0)
            
        Returns:
            Predicted anatomical joint positions [frames, 24, 3]
        """
        # Load regressor
        regressor_file = os.path.join(regressor_path, f"smpl_to_osim_regressor_{gender}.pkl")
        with open(regressor_file, 'rb') as f:
            regressor_data = pickle.load(f)
        
        regressors = regressor_data['regressors']
        scaler_info = regressor_data['scalers']


        # Ensure we have 24 joints as expected by the regressor (pad if necessary)
        num_frames, num_joints, dims = smpl_joints.shape
        if num_joints < 24:
            # Pad with zeros to reach 24 joints
            padding = np.zeros((num_frames, 24 - num_joints, dims))
            smpl_joints = np.concatenate([smpl_joints, padding], axis=1)
        elif num_joints > 24:
            # Truncate to 24 joints
            smpl_joints = smpl_joints[:, :24, :]
        
        # Normalize input joints
        smpl_joints_norm = (smpl_joints - scaler_info['smpl_mean']) / (scaler_info['smpl_std'] + 1e-8)
        
        # Reshape for regressor [frames, 24*3]
        X = smpl_joints_norm.reshape(num_frames, -1)
        
        # Predict anatomical joints
        pred_joints = np.zeros((num_frames, 24, 3))
        for joint_idx, joint_regressors in enumerate(regressors):
            for coord_idx, model in enumerate(joint_regressors):
                pred_joints[:, joint_idx, coord_idx] = model.predict(X)
        
        # Denormalize predictions
        pred_joints = pred_joints * scaler_info['osim_std'] + scaler_info['osim_mean']
        
        # Save if requested
        if output_file:
            np.save(output_file, pred_joints)
            print(f"Saved predictions to {output_file}")
        
        return pred_joints

if __name__ == "__main__":
    # Use male by default, but you can also set to "female"
    output_dir = "Aurel/skeleton_fitting/output/regressor"
    gender = "male"
    regressor = SparseSMPLtoAnatomicalRegressor(output_dir=output_dir, gender=gender)
    
    # Start with a small number of samples to test
    # Once everything works, you can increase this to 10,000 or more
    #regressor.run_pipeline(num_samples=100000, alpha=0.0005)  

    # # Apply regressor to motion sequence
    smpl_file = "Aurel/skeleton_fitting/dno_example_output_save/samples_000500000_avg_seed20_a_person_jumping/trajectory_editing_dno/results.npy"
    output_file = os.path.join(output_dir, "regressed_joints.npy")
    
    # Process trial 0 by default
    trial_idx = 0
    anatomical_joints = regressor.predict_anatomical_joints(
        smpl_file, 
        output_dir, 
        gender="male", 
        trial=trial_idx,
        output_file=output_file
    )
    
    print(f"Processed trial {trial_idx}. Output shape: {anatomical_joints.shape}")