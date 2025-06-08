#!/bin/bash

# add the remaining dependencies
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
conda install anaconda::ffmpeg

# install skel
cd external
git clone https://github.com/MarilynKeller/SKEL.git
cd SKEL
pip install git+https://github.com/mattloper/chumpy 
pip install -e .
cd ..

# install aitviewer-skel
git clone https://github.com/MarilynKeller/aitviewer-skel.git
cd aitviewer-skel
pip install -e .
cd ../../

# move skel files to their new locations
cp -r submodule_files/models external/SKEL/
cp submodule_files/skel_model.py external/SKEL/skel/
cp submodule_files/utils.py external/SKEL/skel/

# move ait files to their new locations
cp submodule_files/load_SKEL_with_real_contact.py external/aitviewer-skel/examples/
cp submodule_files/load_SKEL_with_dynamics_analysis.py external/aitviewer-skel/examples/
cp submodule_files/aitvconfig.yaml external/aitviewer-skel/aitviewer/