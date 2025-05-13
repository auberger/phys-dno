# Aurel - SKEL Integration for Physics-DNO

This folder contains utilities for integrating the SKEL (Skinned Kinematic Estimation of Limbs) model with Physics-DNO to create anatomically accurate skeletal animations from motion data.

## Setup

Before using the scripts in this folder, you need to set up the SKEL environment:

```bash
# Activate conda environment you created for DNO
conda activate gmd

# Navigate to the SKEL directory
cd external/SKEL

# Update pip
pip install -U pip   

# Install chumpy from GitHub (required for SKEL)
pip install git+https://github.com/mattloper/chumpy 

# Install SKEL in development mode
pip install -e .
```

## Note

environment_gmd_mac.yml is the adapted conda env file for Apple Silicon (M1/M2) Macs. Use the provided environment file if needed:

```bash
conda env create -f environment_gmd_mac.yml
```

## Directory Structure

### skeleton_fitting/

This directory contains scripts for fitting skeletal models to motion data:

- **main.py**: Entry point script that processes motion data, creates animations, and regresses SKEL joints.

- **utils/**
  - `animation.py`: Functions for creating animations from motion data.
  - `regress_joints.py`: Tools for regressing SKEL joints from motion data.
  - `joints_utils.py`: Utility functions for joint manipulation and processing.
  - `visualization.py`: Visualization tools for displaying skeletal animations.
  - `calc_dist.py`: Functions for calculating distances between joints.

- **setup/**
  - Contains OpenSim model files (`LaiUhlrich2022.osim`, `LaiUhlrich2022_scaled.osim`)
  - Setup files for inverse kinematics (`IK_setup.xml`) and scaling (`scale_setup.xml`)
  - Marker definitions and configurations for the OpenSim model

## Usage

The main workflow is defined in `skeleton_fitting/main.py` and consists of:

1. Loading motion data from DNO output
2. Running the animation workflow to visualize joint movements
3. Regressing SKEL joints to create anatomically accurate skeletal models
4. (Optional) Converting NPY motion data to TRC format for further analysis