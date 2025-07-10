#!/bin/bash

# Cloud Setup Script for YOLOv13 Triple Input
# This script sets up the environment for cloud deployment

set -e  # Exit on any error

echo "🚀 YOLOv13 Triple Input - Cloud Setup"
echo "===================================="

# Function to print colored output
print_status() {
    echo -e "\e[32m✅ $1\e[0m"
}

print_error() {
    echo -e "\e[31m❌ $1\e[0m"
}

print_info() {
    echo -e "\e[34mℹ️  $1\e[0m"
}

# Check if we're in the right directory
if [ ! -f "standalone_train.py" ]; then
    print_error "standalone_train.py not found. Make sure you're in the project root."
    exit 1
fi

print_info "Setting up Python environment..."

# Update pip
python3 -m pip install --upgrade pip

# Install required packages
print_info "Installing required packages..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install ultralytics
python3 -m pip install opencv-python
python3 -m pip install numpy
python3 -m pip install pyyaml
python3 -m pip install tqdm
python3 -m pip install matplotlib
python3 -m pip install pillow
python3 -m pip install scipy

print_status "Packages installed successfully"

# Set up Python path
export PYTHONPATH="${PWD}:${PWD}/yolov13:${PWD}/yolov13/ultralytics:$PYTHONPATH"
print_info "Python path configured: $PYTHONPATH"

# Make scripts executable
chmod +x standalone_train.py
print_status "Made standalone_train.py executable"

# Verify installation
print_info "Verifying installation..."
python3 -c "
import torch
import cv2
import numpy as np
import yaml
import tqdm
print('✅ Core packages imported successfully')

try:
    from ultralytics import YOLO
    print('✅ Ultralytics imported successfully')
except ImportError as e:
    print(f'⚠️  Ultralytics import warning: {e}')
    print('   (This will be handled by standalone_train.py)')
"

print_status "Setup completed successfully!"

echo ""
echo "🎯 Ready to train! Use these commands:"
echo ""
echo "# Quick test (3 epochs):"
echo "python3 standalone_train.py --data working_dataset.yaml --model s --epochs 3 --batch 1 --device cpu"
echo ""
echo "# Full training:"
echo "python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu"
echo ""
echo "# For GPU (if available):"
echo "python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 16 --device 0"
echo ""

# Create a simple run script
cat > run_training.sh << 'EOF'
#!/bin/bash
# Simple training runner

echo "🚀 Starting YOLOv13 Training..."

# Set environment
export PYTHONPATH="${PWD}:${PWD}/yolov13:${PWD}/yolov13/ultralytics:$PYTHONPATH"

# Run training
python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu

echo "✅ Training completed!"
EOF

chmod +x run_training.sh
print_status "Created run_training.sh for easy execution"

echo ""
print_status "Cloud setup complete! You can now run training with:"
echo "   ./run_training.sh"