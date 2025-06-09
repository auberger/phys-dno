# Diffusion-Noise-Optimization with Differentiable Dynamics Modelling

[![arXiv](https://img.shields.io/badge/arXiv-<2312.11994>-<COLOR>.svg)](https://arxiv.org/abs/2312.11994)


![teaser](./assets/teaser.jpg)


## Getting started

This code works with the basic DNO conda environment. All additions to the environment have been added to the `SETUP.sh` script. This will clone two additional repositories crucial to the forward kinematics calculation and results visualization and install their dependencies automatically. The rest of the setup does not vary from that of DNO so the following commands are largely unchanged. Only the first **Text to Motion** dependency is crucial for running DNO.


### 1. Install dependencies

Setup conda env:

```shell
conda env create -f environment_gmd.yml
conda activate gmd
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download submodules:

```bash
bash SETUP.sh
```

Download dependencies:


<summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```



### 2. Get data

<!-- <details>
  <summary><b>Text to Motion</b></summary> -->

There are two paths to get the data:

(a) **Generation only** wtih pretrained text-to-motion model without training or evaluating

(b) **Get full data** to train and evaluate the model.


#### a. Generation only (text only)

**HumanML3D** - Clone [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D Diffusion-Noise-Optimization/dataset/HumanML3D
cd Diffusion-Noise-Optimization
```


#### b. Full data (text + motion capture)


**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:



Then copy the data to our repository
```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

### 3. Download the pretrained models

Download our version of MDM, then unzip and place it in `./save/`. 
The model is trained on the HumanML3D dataset.

[MDM model with EMA](https://polybox.ethz.ch/index.php/s/ZiXkIzdIsspK2Lt)


## Motion Editing


- To enable or disable different differentiable dynamics loss terms (for example enabling/disbling angular dynamics or simply to revert back to baseline DNO) the corresponding boolean flags can be set in `sample/condition.py`:
  ```python
  com_term = True
  angular_term = False
  ```

- To run a demo, you can simply run a provided motion generation script, for example `motion_gen_jumping.sh`. To get the correct pose targets, the values within the script can be copied into `sample/dno_helper.py`:
  ```python
  def task_pose_editing(task_info, args, target, target_mask):
    target_edit_list = [
          # (joint_index, keyframe, edit_dim, target(x, y, z))
          (0, 20, [0], [-0.2192]), 
          (0, 20, [2], [0.8807]), 
          (0, 40, [0], [-1.1151]),
          (0, 40, [2], [1.5184])
      ]
  ```


# Biomechanical Physics Pipeline (`inverse_dynamics/`)

This directory implements a complete physics validation pipeline that transforms SMPL motion sequences into biomechanically accurate skeletal motion with force analysis. The pipeline consists of four main stages:

## Pipeline Overview

1. **SMPL → Anatomical Joint Mapping** (Sparse Regression)
2. **Inverse Kinematics Fitting** ("SKELify")  
3. **Biomechanical Modeling** (Center of Mass & Inertia)
4. **Contact Forces & Dynamics Validation**

## Core Files

### Main Pipeline
- **`inverse_kinematics.py`** - Main entry point that orchestrates the complete pipeline
  - Function: `run_ik(input_joints, debug=False)` - Processes SMPL joints through full biomechanical analysis
  - Returns: physics loss metrics, contact forces, and SKEL fitting results

### Training & Testing
- **`train.py`** - Trains the sparse linear regressor that maps SMPL joints to anatomical joint positions
  - Uses stratified pose sampling and Lasso regression with anatomical locality constraints
  - Achieves <1cm RMSE with >89% sparsity
- **`test.py`** - Quick test script to validate pipeline functionality and compute physics losses

### Visualization
- **`visualize.py`** - Creates detailed biomechanical visualizations including:
  - Ground reaction forces (GRF) at center of pressure
  - Center of mass trajectory and required forces  
  - Contact spheres and force vectors
  - Configure via `npy_file` and `output_dir` variables

## Directory Structure

### `utils/` - Core Implementation Modules
- **`anatomical_joint_regressor.py`** - Sparse linear regression from SMPL to anatomical joints
- **`anatomical_joint_ik_adam.py`** - Inverse kinematics optimizer for SKEL model fitting
- **`center_of_mass_calculator.py`** - Biomechanical properties calculation (CoM, inertia, angular momentum)
- **`contact_models_torch.py`** - Ground contact detection and force estimation using Kelvin-Voigt model
- **`losses.py`** - Physics validation losses implementing Newton's 2nd law and Euler's equations
- **`joints_utils.py`** - Joint manipulation and coordinate transformation utilities
- **`visualization.py`** - Plotting and animation utilities for biomechanical analysis
- **`animation.py`** - Advanced animation rendering for motion sequences
- **`body_properties_output.py`** - Anthropometric data and segment properties
- **`calc_dist.py`** - Distance and geometric calculations

### `regressor/` - Trained Models & Data
- **`smpl_to_osim_regressor_male.pt`** - Pre-trained sparse regression model
- **`*_data_male.npy`** - Training data for SMPL and anatomical joint correspondences
- **`test_*.npy`** - Validation datasets and predictions

### `output/` - Generated Results
- Stores intermediate results, fitted models, and analysis outputs
- Contains visualizations, fitted poses, and physics validation metrics

## Quick Start

### Basic Pipeline Usage
```python
from inverse_dynamics.inverse_kinematics import run_ik
import torch

# Load SMPL motion data (shape: [frames, joints, 3])
smpl_joints = torch.load("path/to/smpl_motion.pt")

# Run complete physics pipeline
losses, contact_forces, skel_results = run_ik(
    input_joints=smpl_joints,
    debug=True  # Enable detailed logging
)

print(f"Linear dynamics loss: {losses['translational_loss']}")
print(f"Angular dynamics loss: {losses['rotational_loss']}")
```

### Training New Regressor
```python
from inverse_dynamics.utils.anatomical_joint_regressor import SparseSMPLtoAnatomicalRegressor

regressor = SparseSMPLtoAnatomicalRegressor(
    output_dir="./inverse_dynamics/regressor",
    gender="male"
)
regressor.run_pipeline(
    num_samples=20000,
    alpha=0.0005,      # Lasso regularization strength
    learning_rate=0.005,
    num_epochs=2000
)
```

### Visualization Setup
Edit variables in `inverse_dynamics/visualize.py`:
```python
npy_file = "save/your_motion_results/results.npy"
output_dir = "./inverse_dynamics/output"
```

Then generate biomechanical visualizations:
```bash
python inverse_dynamics/visualize.py
```

## Physics Validation Metrics

The pipeline computes two key physics consistency measures:

### Linear Dynamics Loss
Validates Newton's 2nd law: `F_GRF = m(a_COM + g)`
- Compares ground reaction forces with required forces at center of mass
- Lower values indicate better force balance

### Angular Dynamics Loss  
Validates Euler's equation: `M_GRF + M_gravity = dL/dt`
- Compares ground reaction moments with angular momentum rate of change
- Accounts for rotational consistency

## Advanced Visualization (3D Meshes)

For high-quality mesh rendering with surface geometry, use [aitviewer-skel](https://github.com/MarilynKeller/aitviewer-skel).

![SKEL Visualization with Contact Spheres](./assets/skel_with_contact.gif)

### Visualization Scripts

#### Contact Forces Visualization
```bash
python external/aitviewer-skel/examples/load_SKEL_with_real_contact.py \
    -s '/path/to/jumping_ik_results.pkl' \
    -c '/path/to/contact_output.pt' \
    --force_scale 0.002 --sphere_radius 0.032
```

#### Complete Dynamics Analysis
```bash
python external/aitviewer-skel/examples/load_SKEL_with_dynamics_analysis.py \
    -s '/path/to/jumping_ik_results.pkl' \
    -c '/path/to/contact_output.pt' \
    -m '/path/to/com_analysis_results.json' \
    --force_scale 0.0006
```

## Visualization Components

### Linear Dynamics Analysis
Implements Newton's second law: `F_GRF = m(a_COM + g)`

![Linear Dynamics Animation](./assets/linear_dynamics_animation.gif)

- **Blue vectors:** Ground reaction forces at center of pressure
- **Green vectors:** Required forces at center of mass
- **Red sphere:** Center of mass with trajectory trail
- **Orange square:** Center of pressure

### Angular Dynamics Analysis
Implements Euler's equation: `M_GRF + M_gravity = dL/dt`

![Angular Dynamics Animation](./assets/angular_dynamics_animation.gif)

- **Blue vectors:** Moment from ground reaction forces
- **Magenta vectors:** Moment from gravity
- **Green vectors:** Rate of change of angular momentum

This biomechanical pipeline enables detailed physics validation for motion generation algorithms and provides tools for understanding human movement dynamics in applications ranging from animation to clinical biomechanics. 
