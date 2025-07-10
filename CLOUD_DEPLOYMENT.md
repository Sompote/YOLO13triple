# ☁️ YOLOv13 Triple Input - Cloud Deployment Guide

This guide provides step-by-step instructions for deploying YOLOv13 Triple Input on various cloud platforms with standalone execution.

## 🚀 Quick Start (Any Cloud Platform)

### Method 1: Direct Python Execution

```bash
# 1. Clone the repository
git clone https://github.com/Sompote/yolov13-triple-input.git
cd yolov13-triple-input

# 2. Run setup script
chmod +x cloud_setup.sh
./cloud_setup.sh

# 3. Start training
python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu
```

### Method 2: Using Docker

```bash
# 1. Build Docker image
docker build -f Dockerfile.cloud -t yolov13-triple .

# 2. Run training
docker run -v $(pwd)/runs:/workspace/runs yolov13-triple \
    python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu
```

## 🌐 Platform-Specific Instructions

### Google Colab

```python
# Install in Colab notebook
!git clone https://github.com/Sompote/yolov13-triple-input.git
%cd yolov13-triple-input

# Install dependencies
!pip install torch torchvision torchaudio
!pip install ultralytics opencv-python numpy pyyaml tqdm matplotlib pillow scipy

# Run training
!python standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 16 --device 0
```

### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 20.04+ recommended)
# 2. Install Python and Git
sudo apt update
sudo apt install -y python3 python3-pip git

# 3. Clone and setup
git clone https://github.com/Sompote/yolov13-triple-input.git
cd yolov13-triple-input
chmod +x cloud_setup.sh
./cloud_setup.sh

# 4. Train (use screen for long training)
screen -S yolo_training
python3 standalone_train.py --data working_dataset.yaml --model s --epochs 100 --batch 16 --device cpu
# Ctrl+A, D to detach
```

### Google Cloud Platform (GCP)

```bash
# 1. Create Compute Engine instance
gcloud compute instances create yolo-training \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-4 \
    --boot-disk-size=50GB

# 2. SSH and setup
gcloud compute ssh yolo-training
git clone https://github.com/Sompote/yolov13-triple-input.git
cd yolov13-triple-input
./cloud_setup.sh

# 3. Train
python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu
```

### Azure

```bash
# 1. Create Azure VM (Ubuntu 20.04)
# 2. SSH to VM and run:
sudo apt update && sudo apt install -y python3 python3-pip git
git clone https://github.com/Sompote/yolov13-triple-input.git
cd yolov13-triple-input
./cloud_setup.sh
python3 standalone_train.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu
```

## 🔧 Configuration Options

### Model Variants

```bash
# Nano (fastest, smallest)
python3 standalone_train.py --model n --epochs 100 --batch 32

# Small (recommended)
python3 standalone_train.py --model s --epochs 50 --batch 16

# Medium (higher accuracy)
python3 standalone_train.py --model m --epochs 50 --batch 8

# Large (maximum accuracy)
python3 standalone_train.py --model l --epochs 50 --batch 4

# Extra Large (research)
python3 standalone_train.py --model x --epochs 50 --batch 2
```

### Device Configuration

```bash
# CPU only
--device cpu

# Single GPU
--device 0

# Multiple GPUs
--device 0,1,2,3
```

### Custom Dataset

```bash
# Create your dataset structure:
my_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

# Create dataset config (my_dataset.yaml):
path: /path/to/my_dataset
train: images/train
val: images/val
nc: 10  # number of classes
names:
  0: class1
  1: class2
  # ... etc

# Train with custom dataset
python3 standalone_train.py --data my_dataset.yaml --model s --epochs 100
```

## 📊 Monitoring and Results

### Training Monitoring

```bash
# View training progress (in another terminal)
tail -f runs/standalone_train/yolov13s_standalone/train.log

# Monitor with tensorboard (if installed)
pip install tensorboard
tensorboard --logdir runs/standalone_train/
```

### Results Location

```
runs/standalone_train/yolov13{variant}_standalone/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Latest checkpoint
├── train_batch*.jpg     # Training samples
├── val_batch*.jpg       # Validation samples
├── results.png          # Training curves
└── confusion_matrix.png # Confusion matrix
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, ensure Python path is set:
export PYTHONPATH="${PWD}:${PWD}/yolov13:${PWD}/yolov13/ultralytics:$PYTHONPATH"
```

#### 2. Memory Issues
```bash
# Reduce batch size for limited memory:
python3 standalone_train.py --batch 1 --device cpu

# Or use gradient accumulation:
python3 standalone_train.py --batch 4 --accumulate 4
```

#### 3. Permission Issues
```bash
# Make sure scripts are executable:
chmod +x standalone_train.py cloud_setup.sh
```

#### 4. Dataset Not Found
```bash
# Verify dataset paths in your YAML file:
python3 -c "
import yaml
with open('working_dataset.yaml') as f:
    config = yaml.safe_load(f)
    print('Dataset path:', config.get('path'))
    print('Train path:', config.get('train'))
    print('Val path:', config.get('val'))
"
```

### Performance Optimization

#### CPU Optimization
```bash
# Set optimal number of workers
export OMP_NUM_THREADS=4
python3 standalone_train.py --workers 4

# Use smaller image size for faster training
python3 standalone_train.py --imgsz 416
```

#### GPU Optimization
```bash
# Use mixed precision (if supported)
python3 standalone_train.py --amp

# Increase batch size for better GPU utilization
python3 standalone_train.py --batch 32 --device 0
```

## 📋 Cloud Resource Recommendations

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+
- **Storage**: 10GB+
- **Python**: 3.8+

### Recommended Specifications

| Use Case | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| **Development** | 2-4 cores | 8GB | Optional | 20GB |
| **Small Dataset** | 4-8 cores | 16GB | GTX 1660+ | 50GB |
| **Production** | 8+ cores | 32GB+ | RTX 3080+ | 100GB+ |

### Cloud Instance Recommendations

| Platform | Instance Type | Specs | Use Case |
|----------|---------------|-------|----------|
| **AWS** | t3.large | 2 vCPU, 8GB RAM | Development |
| **AWS** | c5.2xlarge | 8 vCPU, 16GB RAM | Training |
| **GCP** | n1-standard-4 | 4 vCPU, 15GB RAM | Balanced |
| **Azure** | Standard_D4s_v3 | 4 vCPU, 16GB RAM | Training |

## 🔐 Security Best Practices

1. **Use virtual environments**:
   ```bash
   python3 -m venv yolo_env
   source yolo_env/bin/activate
   ```

2. **Don't run as root**:
   ```bash
   # Create non-root user
   sudo useradd -m yolo
   sudo su - yolo
   ```

3. **Secure your data**:
   ```bash
   # Set appropriate permissions
   chmod 600 working_dataset.yaml
   chmod 700 runs/
   ```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Python environment: `python3 --version`
3. Check logs: `cat runs/standalone_train/*/train.log`
4. Create an issue on GitHub with full error logs

## 🎯 Next Steps After Training

1. **Evaluate results**:
   ```bash
   # View training curves
   open runs/standalone_train/yolov13s_standalone/results.png
   ```

2. **Test inference**:
   ```bash
   python3 -c "
   from ultralytics import YOLO
   model = YOLO('runs/standalone_train/yolov13s_standalone/weights/best.pt')
   results = model('path/to/test/image.jpg')
   results[0].show()
   "
   ```

3. **Deploy model**:
   - Export to ONNX for production
   - Create inference API
   - Integrate with your application

Happy training! 🚀