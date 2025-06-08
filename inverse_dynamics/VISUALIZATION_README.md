# Visualization Guide

We provide two visualization approaches for analyzing biomechanical motion sequences with inverse dynamics.

## Basic Visualization (Joint Positions & Dynamics)

### Quick Start

```bash
python inverse_dynamics/visualize.py
```

### Configuration

Edit variables in `visualize.py`:
```python
npy_file = "save/samples_000500000_avg_seed20_a_person_is_jumping/pose_editing_dno/results.npy" # Select case you want to visualize
output_dir = "./inverse_dynamics/output"
trial_idx = 1  # Trial to visualize
```

## Advanced Visualization (3D Meshes)

For high-quality mesh rendering with surface geometry, use [aitviewer-skel](https://github.com/MarilynKeller/aitviewer-skel).

![SKEL Visualization with Contact Spheres](../assets/skel_with_contact.gif)

### Requirements
- Python ≥ 3.9
- aitviewer-skel package

### Setup

1. **Install aitviewer-skel:**
```bash
cd external/
git clone https://github.com/MarilynKeller/aitviewer-skel.git
cd aitviewer-skel
pip install -e .
```

2. **Copy scripts:** Move the two scripts from `inverse_dynamics/aitviewer-skel/` to `external/aitviewer-skel/examples/`

3. **Configure paths** in `aitviewer/aitvconfig.yaml`:
```yaml
skel_models: "/path/to/skel_models_v1.0"
osim_geometry: "/path/to/skel_models_v1.0/Geometry"
```

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

![Linear Dynamics Animation](../assets/linear_dynamics_animation.gif)

- **Blue vectors:** Ground reaction forces at center of pressure
- **Green vectors:** Required forces at center of mass
- **Red sphere:** Center of mass with trajectory trail
- **Orange square:** Center of pressure

### Angular Dynamics Analysis
Implements Euler's equation: `M_GRF + M_gravity = dL/dt`

![Angular Dynamics Animation](../assets/angular_dynamics_animation.gif)

- **Blue vectors:** Moment from ground reaction forces
- **Magenta vectors:** Moment from gravity
- **Green vectors:** Rate of change of angular momentum

This visualization framework enables detailed biomechanical analysis for validating motion generation algorithms and understanding human movement physics. 