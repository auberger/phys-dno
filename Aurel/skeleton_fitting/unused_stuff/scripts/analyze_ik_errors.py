#!/usr/bin/env python
"""
Script to analyze IK marker errors from OpenSim .sto files.
This script specifically calculates the average RMS marker error.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_sto_file(file_path):
    """
    Read an OpenSim .sto file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the .sto file
        
    Returns:
        pandas.DataFrame: Data from the .sto file
    """
    # Read the file with pandas, skipping header lines
    with open(file_path, 'r') as f:
        # Find the line with 'endheader'
        header_end = 0
        for i, line in enumerate(f):
            if 'endheader' in line:
                header_end = i
                break
    
    # Now read the file again, skipping the header lines
    df = pd.read_csv(file_path, skiprows=header_end+1, delim_whitespace=True)
    
    return df

def analyze_ik_errors(sto_file):
    """
    Analyze IK marker errors from an OpenSim .sto file.
    
    Args:
        sto_file (str): Path to the IK marker error .sto file
        
    Returns:
        dict: Dictionary containing error statistics
    """
    # Read the .sto file
    df = read_sto_file(sto_file)
    
    # Calculate statistics
    rms_avg = df['marker_error_RMS'].mean()
    rms_std = df['marker_error_RMS'].std()
    rms_min = df['marker_error_RMS'].min()
    rms_max = df['marker_error_RMS'].max()
    
    max_avg = df['marker_error_max'].mean()
    max_std = df['marker_error_max'].std()
    
    # Create results dictionary
    results = {
        'rms_avg': rms_avg,
        'rms_std': rms_std,
        'rms_min': rms_min,
        'rms_max': rms_max,
        'max_avg': max_avg,
        'max_std': max_std,
        'num_frames': len(df)
    }
    
    return results, df

def plot_errors(df, save_path=None):
    """
    Plot the RMS and max marker errors over time.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the error data
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['marker_error_RMS'], label='RMS Error', color='blue')
    plt.plot(df['time'], df['marker_error_max'], label='Max Error', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('IK Marker Errors')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define path to the .sto file - using relative path from script location
    sto_file = os.path.join(script_dir, "..", "setup", "_ik_marker_errors.sto")
    
    # Check if file exists
    if not os.path.exists(sto_file):
        print(f"Error: File {sto_file} not found. Please provide the correct path.")
        exit(1)
    
    # Analyze the errors
    results, df = analyze_ik_errors(sto_file)
    
    # Print the results
    print("\n===== IK MARKER ERROR ANALYSIS =====")
    print(f"Number of frames: {results['num_frames']}")
    print("\nRMS ERROR:")
    print(f"  Average: {results['rms_avg']:.5f} meters ({results['rms_avg']*1000:.2f} mm)")
    print(f"  Std Dev: {results['rms_std']:.5f} meters ({results['rms_std']*1000:.2f} mm)")
    print(f"  Min: {results['rms_min']:.5f} meters ({results['rms_min']*1000:.2f} mm)")
    print(f"  Max: {results['rms_max']:.5f} meters ({results['rms_max']*1000:.2f} mm)")
    
    print("\nMAX ERROR:")
    print(f"  Average: {results['max_avg']:.5f} meters ({results['max_avg']*1000:.2f} mm)")
    print(f"  Std Dev: {results['max_std']:.5f} meters ({results['max_std']*1000:.2f} mm)")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save the figure
    plot_path = os.path.join(output_dir, "ik_marker_errors.png")
    plot_errors(df, save_path=plot_path)
    print(f"\nPlot saved to {plot_path}") 