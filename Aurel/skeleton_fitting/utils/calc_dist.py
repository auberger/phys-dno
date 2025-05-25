import numpy as np
import pandas as pd
import os
import json

def calc_dist(point1, point2):
    """
    Calculate the L2 (Euclidean) distance between two points.
    
    Args:
        point1: First point as array-like object with coordinates [x, y, z]
        point2: Second point as array-like object with coordinates [x, y, z]
        
    Returns:
        float: The Euclidean distance between the two points
    """
    # Convert inputs to numpy arrays for robust handling
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    distance = np.linalg.norm(p2 - p1)
    
    return distance

def parse_point(point_str):
    """Parse a point from string format '(-0.056276 0.85151 0.07726)'"""
    # Remove parentheses and split by spaces
    coords = point_str.strip('()').split()
    # Convert to float
    return [float(coord) for coord in coords]

def calculate_joint_distances(joints):
    """
    Calculate the distances between specific joint centers for each frame and store in a DataFrame.

    Args:
        joints: Motion data with shape (seq_len, num_joints, 3)

    Returns:
        distances_df: A DataFrame containing distances for each frame and segment.
    """
    num_frames, num_joints, _ = joints.shape
    distances = []

    # Define the specific joint pairs to measure
    joint_pairs = [
        (3, 1, "Left_Pelvis_Height"),
        (3, 2, "Right_Pelvis_Height"),
        (1, 2, "Pelvis_Width"),
        (1, 4, "Left_Thigh"),
        (2, 5, "Right_Thigh"),
        (4, 7, "Left_Shank"),
        (5, 8, "Right_Shank"),
        (7, 10, "Left_Foot"),
        (8, 11, "Right_Foot"),
        (3, 16, "Left_Torso_Height"),
        (3, 17, "Right_Torso_Height"),
        (16, 17, "Torso_Width"),
        (16, 18, "Left_Upper_Arm"),
        (17, 19, "Right_Upper_Arm"),
        (18, 20, "Left_Lower_Arm"),
        (19, 21, "Right_Lower_Arm")
    ]
    
    # Calculate distances for each frame
    for frame in range(num_frames):
        frame_data = joints[frame]
        frame_distances = {}

        # Calculate distances for specified joint pairs
        for i, j, name in joint_pairs:
            if i < num_joints and j < num_joints:  # Check that joint indices are valid
                dist = np.linalg.norm(frame_data[i] - frame_data[j])
                frame_distances[name] = dist
            else:
                print(f"Warning: Joint pair ({i},{j}) - {name} includes indices outside range of joints (0-{num_joints-1})")

        distances.append(frame_distances)

    # Create a DataFrame from the list of dictionaries
    distances_df = pd.DataFrame(distances)
    
    # Add a frame number column
    distances_df.insert(0, "Frame", range(num_frames))
    
    # Calculate statistics for each segment
    stats = distances_df.describe()
    print("\nSegment Length Statistics:")
    print(stats)
    
    # Calculate coefficient of variation (std/mean) to check constancy
    cv = stats.loc["std"] / stats.loc["mean"]
    print("\nCoefficient of Variation (lower means more constant):")
    print(cv)
    
    return distances_df

def save_distances_to_json(joint_distances_df, npy_file, json_path="custom/SMPL_distances.json", osim_csv_path="custom/distances_osim.csv"):
    """
    Save joint distances to a JSON file with optional OpenSim comparison.
    
    Args:
        joint_distances_df: DataFrame containing joint distances
        npy_file: Path to the NPY file that was processed
        json_path: Path to save the JSON file
        osim_csv_path: Path to the OpenSim distances CSV file
        
    Returns:
        dict: The created distance data
    """
    # Extract filename
    filename = npy_file.split(".")[0]
    
    # Calculate average distances to include in JSON
    avg_distances = joint_distances_df.drop("Frame", axis=1).mean().to_dict()

    # Create distances dictionary for this file
    file_distances = {
        "filename": filename,
        "num_frames": len(joint_distances_df),
        "average_distances": avg_distances,
        "statistics": {
            "min": joint_distances_df.drop("Frame", axis=1).min().to_dict(),
            "max": joint_distances_df.drop("Frame", axis=1).max().to_dict(),
            "std": joint_distances_df.drop("Frame", axis=1).std().to_dict(),
            "cv": (joint_distances_df.drop("Frame", axis=1).std() / 
                    joint_distances_df.drop("Frame", axis=1).mean()).to_dict()
        }
    }
    
    # Load the generic OpenSim distances from CSV
    if os.path.exists(osim_csv_path):
        osim_distances_df = pd.read_csv(osim_csv_path)
        
        # Extract the first row (assuming it contains the reference values)
        osim_distances = osim_distances_df.iloc[0].to_dict()
        
        # Remove the Frame column if it exists
        if "Frame" in osim_distances:
            del osim_distances["Frame"]
            
        # Add OpenSim distances to our JSON file
        file_distances["opensim_distances"] = osim_distances
        
        # Calculate scale factors (SMPL/OpenSim)
        scale_factors = {}
        for joint_pair, smpl_dist in avg_distances.items():
            if joint_pair in osim_distances:
                opensim_dist = osim_distances[joint_pair]
                if opensim_dist > 0:  # Avoid division by zero
                    scale_factors[joint_pair] = smpl_dist / opensim_dist
                else:
                    scale_factors[joint_pair] = None
            else:
                print(f"Warning: {joint_pair} not found in OpenSim distances")
        
        # Add scale factors to the JSON
        file_distances["scale_factors"] = scale_factors
    else:
        print(f"Warning: OpenSim distances file not found at {osim_csv_path}")
    
    # Check if JSON file exists
    SMPL_distances = {}
    if os.path.exists(json_path):
        # Load existing data
        try:
            with open(json_path, "r") as json_file:
                SMPL_distances = json.load(json_file)
        except json.JSONDecodeError:
            # If file exists but is empty or invalid
            print(f"Creating new JSON file at {json_path}")
    
    # Add or update the entry for this file
    SMPL_distances[filename] = file_distances
    
    # Save the updated data back to the JSON file
    with open(json_path, "w") as json_file:
        json.dump(SMPL_distances, json_file, indent=4, sort_keys=True)
    
    print(f"Saved joint distances to {json_path} with key {filename}")
    
    return file_distances

# Example usage
if __name__ == "__main__":
    # Example with the specified point format: (-0.056276 0.85151 0.07726)
    point_a_str = "(-0.106385 1.38137 -0.17345)"
    point_b_str = "(-0.109673 1.38448 0.16652)"
    
    # Parse the points from string format
    point_a = parse_point(point_a_str)
    point_b = parse_point(point_b_str)
    
    # Calculate distance
    dist_3d = calc_dist(point_a, point_b)
    
    # Print results
    print(f"Distance between points:")
    print(f"  Point A: {point_a_str}")
    print(f"  Point B: {point_b_str}")
    print(f"  Distance: {dist_3d:.6f}")
    

        