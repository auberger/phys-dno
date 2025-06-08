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


# Visualization Guide

We provide two visualization approaches for analyzing biomechanical motion sequences with inverse dynamics.

## Basic Visualization (Joint Positions & Dynamics)

### Configuration

Edit variables in `visualize.py`:
```python
npy_file = "save/seed52_4758_a_person_is_doing_a_squat/results.npy"
output_dir = "./inverse_dynamics/output"
```

### Create output

```bash
python inverse_dynamics/visualize.py
```



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

This visualization framework enables detailed biomechanical analysis for validating motion generation algorithms and understanding human movement physics. 