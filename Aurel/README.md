# Aurel - SKEL Integration for Physics-DNO

This folder contains utilities for integrating the SKEL (Skinned Kinematic Estimation of Limbs) model with Physics-DNO to create anatomically accurate skeletal animations from motion data.

## Setup

This code was tested on MacOS 15.1 (Arm architecture) and requires:

* Python 3.9.22
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.


### 2. Install dependencies

DNO uses the same dependencies as GMD so if you already install one, you can use the same environment here.

Setup conda env:

```shell
conda env create -f environment_gmd_updated.yml
conda activate gmd_updated
conda remove --force ffmpeg
conda install anaconda::ffmpeg
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:


<summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

Before using the scripts in this folder, you need to set up the SKEL environment:

```bash
# Navigate to the SKEL directory
cd external/SKEL

# Update pip
pip install -U pip   

# Install chumpy from GitHub (required for SKEL)
pip install git+https://github.com/mattloper/chumpy 

# Install SKEL in development mode
pip install -e .

# [OPTIONAL] Create other env for visualizations only (requires python >3.10); we want to avoid compatibility issues
# Navigate to the Aitviewer dir (used for visualization)
cd ..
cd aitviewer-skel

# Install Aitviewer in development mode
pip install -e .

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