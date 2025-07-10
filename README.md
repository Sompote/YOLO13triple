# YOLOv13 Multi-Variant Training System

A clean, optimized implementation of YOLOv13 with complete support for all model variants and PyTorch-compatible training. **Cloud-deployment ready with self-contained local dependencies.**

## 🚀 Quick Start

### Installation
```bash
# Install PyTorch-compatible packages (cloud-ready)
pip install -r requirements.txt

# Test installation and local dependencies
python test_package_stability.py
python test_local_import.py
```

### ☁️ Cloud Deployment
This repository is designed for cloud deployment with **local-only dependencies**:
- ✅ All YOLO/ultralytics code is included locally in `yolov13/` directory
- ✅ No external ultralytics package installation required
- ✅ Self-contained with fixed NumPy compatibility
- ✅ Works in containerized environments (Docker, cloud platforms)

### Simple Training
```bash
# Train with any model variant (n, s, m, l, x)
python simple_train.py --data working_dataset.yaml --epochs 50 --batch 4 --device cpu --variant s

# Examples for different variants
python simple_train.py --data working_dataset.yaml --variant n --epochs 50 --batch 8  # Nano (fastest)
python simple_train.py --data working_dataset.yaml --variant s --epochs 50 --batch 4  # Small (balanced)
python simple_train.py --data working_dataset.yaml --variant m --epochs 50 --batch 2  # Medium (better)
python simple_train.py --data working_dataset.yaml --variant l --epochs 50 --batch 1  # Large (best)
```

## 📊 Model Variants

| Variant | Size | Parameters | Speed | Memory | Recommended Use |
|---------|------|------------|-------|--------|-----------------|
| **n** | Nano | ~3M | Fastest | Lowest | Real-time detection |
| **s** | Small | ~7M | Fast | Low | General purpose |
| **m** | Medium | ~21M | Medium | Medium | High accuracy |
| **l** | Large | ~47M | Slow | High | Maximum accuracy |
| **x** | Extra-Large | ~86M | Slowest | Highest | Research/benchmarks |

## 🎯 Key Features

### ✅ **Complete Compatibility**
- **PyTorch Compatible**: Uses NumPy < 2.0 for full compatibility
- **Cloud-Ready**: Self-contained local dependencies, no external YOLO/ultralytics packages needed
- **All Model Variants**: Supports n, s, m, l, x variants
- **Stable Training**: Optimized configurations prevent errors
- **Auto-Detection**: Prevents NumPy 2.x compatibility issues

### 🚀 **Optimized Training**
- **Smart Batch Sizing**: Automatic adjustment for model size
- **Stable Augmentations**: Disabled problematic augmentations
- **Memory Efficient**: Optimized for resource usage
- **Error Handling**: Comprehensive error recovery

### 🔧 **Easy to Use**
- **Simple Commands**: One-line training for any variant
- **Flexible Options**: Customizable training parameters
- **Test Scripts**: Built-in compatibility verification
- **Clear Documentation**: Comprehensive usage guide

## 📁 Repository Structure

```
yolo_3dual_input/
├── 🎯 Training Scripts
│   ├── simple_train.py              # ⭐ Recommended - All variants
│   ├── standalone_train_fixed.py    # Enhanced training
│   └── train_triple_fixed.py        # Triple input support
├── 🧪 Testing & Verification
│   ├── test_package_stability.py    # Package compatibility check
│   └── test_local_import.py         # Import verification
├── 📊 Configuration
│   ├── working_dataset.yaml         # Dataset configuration
│   ├── triple_dataset.yaml          # Triple input dataset
│   └── yolov13s_standalone.yaml     # Standalone model config
├── 🔧 Setup & Dependencies
│   ├── requirements.txt             # PyTorch-compatible requirements
│   ├── setup.py                     # Package setup
│   └── .gitignore                   # Git ignore rules
├── 📖 Documentation
│   ├── README_FIXED_TRAINING.md     # Training troubleshooting
│   ├── README_triple_input.md       # Triple input guide
│   └── README_detection.md          # Detection examples
├── 🎨 Examples & Demos
│   ├── examples/                    # Usage examples
│   ├── triple_inference.py          # Triple input inference
│   └── detect_triple.py             # Detection script
└── 📂 YOLOv13 Core
    └── yolov13/                     # ⭐ Local ultralytics implementation (cloud-ready)
```

## 🔧 Training Scripts

### 1. `simple_train.py` ⭐ **RECOMMENDED**

**Best for:** General training with all model variants

```bash
# Basic usage
python simple_train.py --data working_dataset.yaml --variant s --epochs 50

# Advanced options
python simple_train.py \
    --data working_dataset.yaml \
    --variant m \
    --epochs 100 \
    --batch 4 \
    --device cpu
```

**Features:**
- ✅ All model variants (n, s, m, l, x)
- ✅ PyTorch compatibility verification
- ✅ Automatic batch size adjustment
- ✅ NumPy 2.x protection
- ✅ Stable training configuration

### 2. `standalone_train_fixed.py`

**Best for:** Enhanced training with advanced features

```bash
# Usage
python standalone_train_fixed.py \
    --data working_dataset.yaml \
    --model s \
    --epochs 50 \
    --batch 4 \
    --device cpu
```

**Features:**
- ✅ All model variants
- ✅ Automatic dependency installation
- ✅ Enhanced error handling
- ✅ PyTorch-compatible package management

### 3. `train_triple_fixed.py`

**Best for:** Triple input configuration

```bash
# Usage
python train_triple_fixed.py --data triple_dataset.yaml --epochs 100
```

**Features:**
- ✅ Triple input support
- ✅ Advanced configuration options
- ✅ Custom architecture support

## 🧪 Testing & Verification

### Package Stability Test
```bash
# Check compatibility before training
python test_package_stability.py
```

**What it checks:**
- NumPy version compatibility
- PyTorch-NumPy integration
- Package versions
- Model configurations
- Training script availability

### Import Test
```bash
# Test ultralytics imports
python test_local_import.py
```

## 💡 Usage Examples

### Quick Training Test
```bash
# Test with nano model (fastest)
python simple_train.py --data working_dataset.yaml --epochs 2 --batch 4 --variant n

# Test with small model (recommended)
python simple_train.py --data working_dataset.yaml --epochs 2 --batch 2 --variant s
```

### Production Training
```bash
# High-speed training (nano)
python simple_train.py --data working_dataset.yaml --epochs 100 --batch 8 --variant n

# Balanced training (small)
python simple_train.py --data working_dataset.yaml --epochs 100 --batch 4 --variant s

# High-accuracy training (large)
python simple_train.py --data working_dataset.yaml --epochs 200 --batch 1 --variant l
```

### GPU Training
```bash
# Single GPU
python simple_train.py --data working_dataset.yaml --epochs 100 --batch 8 --device 0 --variant s

# Multiple GPUs
python simple_train.py --data working_dataset.yaml --epochs 100 --batch 16 --device 0,1 --variant m
```

## 🔍 Troubleshooting

### Package Issues
```bash
# Check current versions
python -c "import numpy, torch; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}')"

# Fix NumPy 2.x issues
pip install "numpy<2.0" --force-reinstall

# Complete reinstall
pip install -r requirements.txt --force-reinstall
```

### Training Issues
```bash
# Memory issues - use smaller batch
python simple_train.py --data working_dataset.yaml --variant n --batch 2

# GPU issues - use CPU
python simple_train.py --data working_dataset.yaml --variant s --device cpu

# Model issues - check configuration
python test_package_stability.py
```

## 📦 Package Management

### Required Versions
```
numpy < 2.0        # PyTorch compatibility
torch >= 2.2.0     # Core ML framework
opencv-python < 4.10  # Computer vision
pillow < 11.0      # Image processing
```

### Installation Commands
```bash
# Standard installation
pip install -r requirements.txt

# Manual installation
pip install "numpy<2.0" "torch>=2.2.0" "opencv-python<4.10" "pillow<11.0"

# Force compatible versions
pip install "numpy<2.0" "opencv-python<4.10" "pillow<11.0" --force-reinstall
```

## 🎯 Performance Optimization

### Batch Size Guidelines
- **Nano (n)**: 4-8 batch size
- **Small (s)**: 4-6 batch size  
- **Medium (m)**: 2-4 batch size
- **Large (l)**: 1-2 batch size
- **Extra-Large (x)**: 1 batch size

### Training Speed Tips
1. **Use appropriate variant**: Start with nano for speed
2. **Optimize batch size**: Larger batches = faster training
3. **Use GPU**: Significantly faster than CPU
4. **Reduce epochs**: For testing, use 10-20 epochs
5. **Disable augmentations**: Already optimized for stability

## 📞 Support

### Common Issues
1. **NumPy 2.x errors**: Run `pip install "numpy<2.0" --force-reinstall`
2. **Memory errors**: Reduce batch size or use smaller variant
3. **Import errors**: Run `python test_local_import.py`
4. **Training failures**: Check `python test_package_stability.py`
5. **Cloud deployment issues**: Ensure `yolov13/` directory is included in deployment

### ☁️ Cloud Deployment Checklist
- ✅ Include entire `yolov13/` directory in deployment
- ✅ Install only requirements.txt dependencies
- ✅ Do NOT install external ultralytics package
- ✅ Test with `python test_local_import.py` after deployment

### Quick Fixes
```bash
# Reset environment
pip install -r requirements.txt --force-reinstall

# Test everything
python test_package_stability.py

# Start fresh training
python simple_train.py --data working_dataset.yaml --variant s --epochs 10
```

## 📈 Next Steps

### Local Development
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test setup**: `python test_package_stability.py`
3. **Test imports**: `python test_local_import.py`
4. **Choose variant**: n(fast) → s(balanced) → m(better) → l(best)
5. **Start training**: `python simple_train.py --data working_dataset.yaml --variant s`

### ☁️ Cloud Deployment
1. **Upload repository**: Include entire repository with `yolov13/` directory
2. **Install dependencies**: `pip install -r requirements.txt` (no additional packages needed)
3. **Test deployment**: `python test_local_import.py`
4. **Run training**: Same commands as local development

---

**🎉 Cloud-ready YOLOv13 with self-contained dependencies and complete variant support!** 

*All training scripts use local ultralytics implementation - no external package conflicts in cloud environments*