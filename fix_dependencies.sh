#!/bin/bash

echo "=== Fixing RunPod Dependencies for YOLOv13 Triple Input ==="

# Step 1: Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Step 2: Fix NumPy version compatibility
echo "Fixing NumPy version compatibility..."
pip uninstall numpy -y
pip install "numpy<2.0" --force-reinstall

# Step 3: Install missing dependencies
echo "Installing missing dependencies..."
pip install huggingface_hub
pip install transformers

# Step 4: Reinstall PyTorch with proper NumPy compatibility
echo "Reinstalling PyTorch with proper dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall

# Step 5: Install additional YOLO dependencies
echo "Installing additional YOLO dependencies..."
pip install ultralytics
pip install opencv-python
pip install pillow
pip install pyyaml
pip install requests
pip install scipy
pip install matplotlib
pip install seaborn
pip install pandas
pip install tqdm

# Step 6: Install specific versions for compatibility
echo "Installing specific compatible versions..."
pip install "numpy>=1.21.0,<2.0"
pip install "torch>=1.13.0"
pip install "torchvision>=0.14.0"

# Step 7: Verify installations
echo "Verifying installations..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import huggingface_hub; print('huggingface_hub installed successfully')"

echo "=== Dependencies fixed! ==="
echo "You can now run your training script." 