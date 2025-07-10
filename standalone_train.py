#!/usr/bin/env python3
"""
Standalone YOLOv13 Triple Input Training Script for Cloud Deployment
This script includes all necessary dependencies and path configurations for remote deployment.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import argparse
import yaml

def setup_environment():
    """Setup the environment for standalone execution"""
    print("🔧 Setting up standalone environment...")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    yolov13_path = current_dir / "yolov13"
    
    # Add paths to Python path
    paths_to_add = [
        str(yolov13_path),
        str(yolov13_path / "ultralytics"),
        str(current_dir),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"✅ Added paths to Python path:")
    for path in paths_to_add:
        print(f"   - {path}")
    
    return current_dir, yolov13_path

def check_dependencies():
    """Check and install required dependencies"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'numpy',
        'pyyaml',
        'tqdm',
        'matplotlib',
        'pillow',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    
    print("✅ All dependencies are available")
    return True

def import_ultralytics():
    """Import ultralytics with fallback mechanisms"""
    print("📥 Importing ultralytics...")
    
    try:
        # Try direct import first
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        print("✅ Successfully imported ultralytics")
        return YOLO, LOGGER, True
    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
        
        # Try installing ultralytics
        print("🔧 Attempting to install ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
            from ultralytics.utils import LOGGER
            print("✅ Successfully installed and imported ultralytics")
            return YOLO, LOGGER, True
        except Exception as e:
            print(f"❌ Failed to install/import ultralytics: {e}")
            return None, None, False

def create_standalone_model_config(variant='s', nc=10):
    """Create a standalone model configuration"""
    print(f"🏗️ Creating standalone YOLOv13{variant} configuration...")
    
    # Model configurations for different variants
    configs = {
        'n': {
            'depth': 0.33,
            'width': 0.25,
            'channels': [32, 64, 128, 256, 512],
            'description': 'Nano variant (fastest, smallest)'
        },
        's': {
            'depth': 0.33, 
            'width': 0.50,
            'channels': [32, 64, 128, 256, 512],
            'description': 'Small variant (balanced speed and accuracy)'
        },
        'm': {
            'depth': 0.67,
            'width': 0.75, 
            'channels': [48, 96, 192, 384, 768],
            'description': 'Medium variant (good balance)'
        },
        'l': {
            'depth': 1.0,
            'width': 1.0,
            'channels': [64, 128, 256, 512, 512],
            'description': 'Large variant (higher accuracy)'
        },
        'x': {
            'depth': 1.0,
            'width': 1.5,
            'channels': [96, 192, 384, 512, 512],
            'description': 'Extra Large variant (highest accuracy)'
        }
    }
    
    config = configs.get(variant, configs['s'])
    
    # Create YOLOv13 configuration
    model_config = {
        'nc': nc,
        'scales': {
            variant: [config['depth'], config['width'], 1024]
        },
        'backbone': [
            [-1, 1, 'Conv', [config['channels'][0], 3, 2]],
            [-1, 1, 'Conv', [config['channels'][1], 3, 2, 1, 2]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][2]]],
            [-1, 1, 'Conv', [config['channels'][2], 3, 2]],
            [-1, int(6 * config['depth']), 'C2f', [config['channels'][3]]],
            [-1, 1, 'Conv', [config['channels'][3], 3, 2]],
            [-1, int(6 * config['depth']), 'C2f', [config['channels'][3]]],
            [-1, 1, 'Conv', [config['channels'][4], 3, 2]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][4]]],
            [-1, 1, 'SPPF', [config['channels'][4], 5]]
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][3]]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][2]]],
            [-1, 1, 'Conv', [config['channels'][2], 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][3]]],
            [-1, 1, 'Conv', [config['channels'][3], 3, 2]],
            [[-1, 9], 1, 'Concat', [1]],
            [-1, int(3 * config['depth']), 'C2f', [config['channels'][4]]],
            [[15, 18, 21], 1, 'Detect', [nc]]
        ]
    }
    
    # Save configuration
    config_path = f"yolov13{variant}_standalone.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    print(f"✅ Created {config_path} ({config['description']})")
    return config_path

def standalone_train(data_config, model_variant='s', epochs=50, batch_size=8, device='cpu'):
    """Standalone training function that works without complex dependencies"""
    print("🚀 Starting Standalone YOLOv13 Training")
    
    # Import ultralytics
    YOLO, LOGGER, success = import_ultralytics()
    if not success:
        print("❌ Cannot proceed without ultralytics")
        return False
    
    # Create standalone model config
    model_config = create_standalone_model_config(model_variant)
    
    try:
        # Try to load existing model file first
        yolov13_dir = Path("yolov13/ultralytics/cfg/models/v13")
        existing_config = yolov13_dir / f"yolov13{model_variant}.yaml"
        
        if existing_config.exists():
            print(f"✅ Using existing model config: {existing_config}")
            model = YOLO(str(existing_config))
        else:
            print(f"✅ Using standalone model config: {model_config}")
            model = YOLO(model_config)
        
        # Training arguments
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'imgsz': 640,
            'batch': batch_size,
            'device': device,
            'project': 'runs/standalone_train',
            'name': f'yolov13{model_variant}_standalone',
            'save': True,
            'verbose': True,
            'patience': 30,
            'lr0': 0.01,
            'warmup_epochs': 3,
            'workers': 0 if device == 'cpu' else 4,
        }
        
        print(f"🎯 Training Configuration:")
        for key, value in train_args.items():
            print(f"   {key}: {value}")
        
        # Start training
        print("🚀 Starting training...")
        results = model.train(**train_args)
        
        print("✅ Training completed successfully!")
        
        # Save model info
        model_path = f"runs/standalone_train/yolov13{model_variant}_standalone/weights/best.pt"
        print(f"💾 Best model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dataset(data_config):
    """Verify dataset exists and is properly formatted"""
    print(f"🔍 Verifying dataset: {data_config}")
    
    if not Path(data_config).exists():
        print(f"❌ Dataset config not found: {data_config}")
        return False
    
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key in dataset config: {key}")
                return False
        
        # Check if dataset paths exist
        if 'path' in config:
            dataset_path = Path(config['path'])
            train_path = dataset_path / config['train']
            val_path = dataset_path / config['val']
            
            if not train_path.exists():
                print(f"❌ Training images not found: {train_path}")
                return False
            
            if not val_path.exists():
                print(f"❌ Validation images not found: {val_path}")
                return False
        
        print(f"✅ Dataset verified: {config['nc']} classes")
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset config: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Standalone YOLOv13 Triple Input Training")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset configuration file")
    parser.add_argument("--model", type=str, default="yolov13s", help="Model variant (yolov13n, yolov13s, yolov13m, yolov13l, yolov13x)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, 1, ...)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 YOLOv13 Triple Input - Standalone Cloud Training")
    print("=" * 60)
    
    # Setup environment
    current_dir, yolov13_path = setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Extract model variant
    model_variant = args.model.replace('yolov13', '') if 'yolov13' in args.model else args.model
    if model_variant not in ['n', 's', 'm', 'l', 'x']:
        print(f"❌ Invalid model variant: {model_variant}. Use n, s, m, l, or x")
        return False
    
    # Verify dataset
    if not verify_dataset(args.data):
        print("❌ Dataset verification failed")
        return False
    
    # Start training
    success = standalone_train(
        data_config=args.data,
        model_variant=model_variant,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    if success:
        print("\n🎉 Standalone training completed successfully!")
        print("\n📊 Results:")
        print(f"   Model: YOLOv13{model_variant}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Device: {args.device}")
        print(f"   Saved: runs/standalone_train/yolov13{model_variant}_standalone/")
    else:
        print("\n❌ Training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)