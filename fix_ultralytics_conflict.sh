#!/bin/bash

echo "=== Fixing Ultralytics Import Conflict for YOLOv13 Triple Input ==="

# Step 1: Remove conflicting ultralytics package
echo "Removing conflicting ultralytics package..."
pip uninstall ultralytics -y

# Step 2: Fix NumPy version compatibility
echo "Fixing NumPy version compatibility..."
pip uninstall numpy -y
pip install "numpy<2.0" --force-reinstall

# Step 3: Install core dependencies without ultralytics
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
pip install pillow
pip install pyyaml
pip install requests
pip install scipy
pip install matplotlib
pip install seaborn
pip install pandas
pip install tqdm
pip install huggingface_hub
pip install transformers
pip install psutil
pip install thop
pip install tensorboard
pip install wandb

# Step 4: Install YOLOv13 dependencies from local requirements
echo "Installing YOLOv13 specific dependencies..."
cd yolov13
pip install -r requirements.txt

# Step 5: Install the local ultralytics in development mode
echo "Installing local ultralytics in development mode..."
pip install -e .

# Step 6: Set up environment variables
echo "Setting up environment variables..."
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Step 7: Verify the installation
echo "Verifying installation..."
cd ..
python -c "import sys; print('Python path:', sys.path[:3])"
python -c "from ultralytics.data.triple_dataset import TripleYOLODataset; print('Triple dataset import successful!')"
python -c "import ultralytics; print('Ultralytics location:', ultralytics.__file__)"

echo "=== Fix completed! ==="
echo "You can now run triple input training." 