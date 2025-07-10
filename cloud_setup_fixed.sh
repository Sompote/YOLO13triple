#!/bin/bash

# Fixed Cloud Setup Script for YOLOv13 Triple Input
# This script resolves common cloud deployment issues

set -e  # Exit on any error

echo "🚀 YOLOv13 Triple Input - Fixed Cloud Setup"
echo "==========================================="

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

print_warning() {
    echo -e "\e[33m⚠️  $1\e[0m"
}

# Check if we're in the right directory
if [ ! -f "standalone_train_fixed.py" ]; then
    print_error "standalone_train_fixed.py not found. Make sure you're in the project root."
    exit 1
fi

print_info "Setting up Python environment with conflict resolution..."

# Update pip
python3 -m pip install --upgrade pip --no-warn-script-location

# Fix NumPy version conflict first
print_info "Fixing NumPy version conflict..."
python3 -m pip install "numpy<2.0" --force-reinstall --no-warn-script-location

# Install PyTorch with compatible versions
print_info "Installing PyTorch with CPU support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location

# Install other required packages with version constraints
print_info "Installing other required packages..."
python3 -m pip install "opencv-python>=4.5.0" --no-warn-script-location
python3 -m pip install "pyyaml>=5.4.0" --no-warn-script-location
python3 -m pip install "tqdm>=4.62.0" --no-warn-script-location
python3 -m pip install "matplotlib>=3.4.0" --no-warn-script-location
python3 -m pip install "pillow>=8.3.0" --no-warn-script-location
python3 -m pip install "scipy>=1.7.0" --no-warn-script-location

# Install ultralytics last to avoid conflicts
print_info "Installing ultralytics..."
python3 -m pip install "ultralytics>=8.0.0" --no-warn-script-location

print_status "Packages installed successfully"

# Set up Python path
export PYTHONPATH="${PWD}:$PYTHONPATH"
print_info "Python path configured: $PYTHONPATH"

# Make scripts executable
chmod +x standalone_train_fixed.py
chmod +x standalone_train.py
print_status "Made scripts executable"

# Verify installation
print_info "Verifying installation..."
python3 -c "
import sys
print('Python version:', sys.version)

try:
    import numpy as np
    print('✅ NumPy version:', np.__version__)
    if np.__version__.startswith('2.'):
        print('⚠️  NumPy 2.x detected - may cause issues')
    else:
        print('✅ NumPy version compatible')
except Exception as e:
    print('❌ NumPy issue:', e)

try:
    import torch
    print('✅ PyTorch version:', torch.__version__)
except Exception as e:
    print('❌ PyTorch issue:', e)

try:
    import cv2
    print('✅ OpenCV version:', cv2.__version__)
except Exception as e:
    print('❌ OpenCV issue:', e)

try:
    from ultralytics import YOLO
    print('✅ Ultralytics imported successfully')
except Exception as e:
    print('⚠️  Ultralytics import warning:', e)
    print('   (This will be handled by the standalone script)')
"

# Create environment activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Environment activation for YOLOv13 Triple Input

export PYTHONPATH="${PWD}:$PYTHONPATH"
echo "🌟 YOLOv13 environment activated"
echo "Python path: $PYTHONPATH"
echo ""
echo "Ready to train! Use:"
echo "  python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu"
EOF

chmod +x activate_env.sh
print_status "Created activate_env.sh"

# Create a simple run script
cat > run_training_fixed.sh << 'EOF'
#!/bin/bash
# Fixed training runner that handles common issues

echo "🚀 Starting Fixed YOLOv13 Training..."

# Set environment
export PYTHONPATH="${PWD}:$PYTHONPATH"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, using device 0"
    DEVICE="0"
    BATCH_SIZE="16"
else
    echo "💻 Using CPU"
    DEVICE="cpu"
    BATCH_SIZE="8"
fi

# Run training with error handling
echo "🎯 Running: python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch $BATCH_SIZE --device $DEVICE"

if python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch $BATCH_SIZE --device $DEVICE; then
    echo "✅ Training completed successfully!"
    echo "📊 Check results in: runs/standalone_train/"
else
    echo "❌ Training failed. Trying with reduced settings..."
    echo "🔄 Retrying with batch size 1 and CPU..."
    python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 10 --batch 1 --device cpu
fi
EOF

chmod +x run_training_fixed.sh
print_status "Created run_training_fixed.sh"

print_status "Fixed cloud setup complete!"

echo ""
echo "🎯 Ready to train! Choose your option:"
echo ""
echo "Option 1 - Fixed script (recommended):"
echo "  python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu"
echo ""
echo "Option 2 - Auto-run script:"
echo "  ./run_training_fixed.sh"
echo ""
echo "Option 3 - Quick test (3 epochs):"
echo "  python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 3 --batch 1 --device cpu"
echo ""

print_info "Troubleshooting tips:"
echo "- If NumPy errors persist: pip install 'numpy<2.0' --force-reinstall"
echo "- If memory issues: reduce --batch to 1"
echo "- If GPU errors: use --device cpu"
echo "- Check logs in: runs/standalone_train/"