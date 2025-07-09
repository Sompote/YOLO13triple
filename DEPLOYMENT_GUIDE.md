# YOLOv13 Triple Input Model - Deployment Guide

This guide provides complete instructions for deploying the YOLOv13 Triple Input Model on any machine as a standalone application.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.9-3.11
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB+ free space

## 🚀 Quick Start (Automated Setup)

### Option 1: Automated Setup Script
```bash
# Download and run the setup script
python setup_deployment.py
```

This will automatically:
- Check system requirements
- Create virtual environment
- Install all dependencies
- Set up directory structure
- Create run scripts
- Verify installation

### Option 2: Manual Setup

#### Step 1: Clone/Download the Repository
```bash
git clone <repository-url>
cd yolo_3dual_input
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv yolo_env

# Activate it
# On Windows:
yolo_env\Scripts\activate
# On macOS/Linux:
source yolo_env/bin/activate
```

#### Step 3: Install Dependencies
```bash
# For full deployment (recommended)
pip install -r requirements-deployment.txt

# For minimal deployment
pip install -r requirements-minimal.txt
```

#### Step 4: Verify Installation
```bash
python demo_verification.py
```

## 📦 Deployment Options

### 1. Local Deployment

#### Using the Run Scripts
```bash
# On Windows
run_demo.bat

# On macOS/Linux
./run_demo.sh
```

#### Manual Execution
```bash
# Activate environment
source yolo_env/bin/activate  # or yolo_env\Scripts\activate on Windows

# Run the demo
python demo_verification.py

# Or run training
python train_triple.py --data triple_dataset.yaml --epochs 10

# Or run detection
python detect_triple.py --primary image1.jpg --detail1 image2.jpg --detail2 image3.jpg
```

### 2. Docker Deployment

#### Prerequisites
- Docker installed
- Docker Compose installed

#### Build and Run
```bash
# Build and start the container
docker-compose up --build

# Or build manually
docker build -t yolo-triple-input .
docker run -p 7860:7860 yolo-triple-input
```

### 3. Cloud Deployment

#### Using Virtual Machines
1. Create VM with the required specifications
2. Follow the manual setup steps
3. Configure firewall rules if needed
4. Run the application

#### Using Container Services
1. Build Docker image
2. Push to container registry
3. Deploy to cloud container service (AWS ECS, Google Cloud Run, etc.)

## 🔧 Configuration

### Environment Variables
```bash
# Set Python path
export PYTHONPATH=$PYTHONPATH:/path/to/yolo_3dual_input:/path/to/yolo_3dual_input/yolov13

# Set device preference
export YOLO_DEVICE=cpu  # or cuda:0 for GPU
```

### Configuration Files
- `config/deployment_config.json`: Main deployment configuration
- `triple_dataset.yaml`: Dataset configuration
- `requirements-*.txt`: Dependency specifications

### Model Configuration
Edit `config/deployment_config.json`:
```json
{
  "model_path": "weights/best.pt",
  "input_size": [640, 640],
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "max_detections": 100,
  "device": "auto",
  "classes": ["person", "bicycle", "car", ...]
}
```

## 📁 Directory Structure

```
yolo_3dual_input/
├── yolov13/                    # YOLOv13 implementation
├── training_data_demo/         # Demo dataset
├── models/                     # Model definitions
├── weights/                    # Trained model weights
├── data/                       # Input data
├── outputs/                    # Output results
├── logs/                       # Log files
├── config/                     # Configuration files
├── requirements-deployment.txt # Full dependencies
├── requirements-minimal.txt    # Minimal dependencies
├── setup_deployment.py         # Automated setup
├── demo_verification.py        # Verification script
├── train_triple.py            # Training script
├── detect_triple.py           # Detection script
└── DEPLOYMENT_GUIDE.md        # This file
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/yolov13
```

#### 2. CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
python detect_triple.py --device cpu
```

#### 3. Memory Issues
```bash
# Reduce batch size
python train_triple.py --batch 1

# Use minimal requirements
pip install -r requirements-minimal.txt
```

#### 4. Permission Issues (Linux/macOS)
```bash
# Make scripts executable
chmod +x run_demo.sh
chmod +x setup_deployment.py
```

### Dependency Issues

#### Missing System Libraries (Linux)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx
```

#### Python Version Issues
```bash
# Check Python version
python --version

# Use specific Python version
python3.9 -m venv yolo_env
```

## 🧪 Testing Deployment

### Verification Checklist
- [ ] Virtual environment created successfully
- [ ] All dependencies installed without errors
- [ ] Demo verification script runs successfully
- [ ] Training data loads correctly
- [ ] Model inference works
- [ ] Output files are generated

### Test Commands
```bash
# Check requirements
python check_requirements.py

# Run verification
python demo_verification.py

# Test training (quick)
python train_triple.py --data triple_dataset.yaml --epochs 1

# Test detection
python detect_triple.py --primary training_data_demo/images/primary/val/image_1.jpg --detail1 training_data_demo/images/detail1/val/image_1.jpg --detail2 training_data_demo/images/detail2/val/image_1.jpg
```

## 📊 Performance Optimization

### CPU Optimization
```bash
# Set number of workers
export OMP_NUM_THREADS=4

# Use CPU-optimized model
python detect_triple.py --device cpu --batch 1
```

### GPU Optimization
```bash
# Use GPU if available
python detect_triple.py --device cuda:0

# Monitor GPU usage
nvidia-smi -l 1
```

### Memory Optimization
```bash
# Reduce image size
python detect_triple.py --imgsz 416

# Use gradient checkpointing
python train_triple.py --gradient-checkpointing
```

## 🔐 Security Considerations

### Input Validation
- Validate image formats and sizes
- Sanitize file paths
- Limit batch sizes to prevent DoS

### Model Security
- Use secure model loading
- Validate model weights
- Implement access controls

### Network Security
- Use HTTPS for web interfaces
- Implement authentication
- Restrict network access

## 📈 Monitoring and Logging

### Application Logs
- Check `logs/` directory for application logs
- Monitor system resource usage
- Track inference performance

### Health Checks
```bash
# Check application health
curl http://localhost:7860/health

# Monitor processes
ps aux | grep python
```

## 🆘 Support and Maintenance

### Getting Help
1. Check this deployment guide
2. Review troubleshooting section
3. Check system requirements
4. Verify installation steps

### Updates
```bash
# Update dependencies
pip install --upgrade -r requirements-deployment.txt

# Update model weights
# Download new weights and replace in weights/ directory
```

### Backup
```bash
# Backup important files
tar -czf backup.tar.gz weights/ config/ data/

# Restore from backup
tar -xzf backup.tar.gz
```

## 📝 License and Credits

- YOLOv13 Implementation
- PyTorch Framework
- OpenCV Library
- Additional open-source libraries

---

**Note**: This deployment guide ensures the YOLOv13 Triple Input Model can be deployed as a standalone application on any compatible machine. Follow the steps carefully for successful deployment.