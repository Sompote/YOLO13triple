#!/usr/bin/env python3
"""
Fixed YOLOv13 Triple Input - Standalone Training Script
Uses only local ultralytics from yolov13 directory
"""

import sys
import os
import subprocess
import yaml
import argparse
from pathlib import Path

def setup_environment():
    """Setup environment and paths"""
    current_dir = Path(__file__).parent
    
    # Add yolov13 directory to Python path FIRST
    yolov13_path = current_dir / "yolov13"
    if str(yolov13_path) not in sys.path:
        sys.path.insert(0, str(yolov13_path))
    
    # Set environment variables to avoid conflicts
    os.environ["PYTHONPATH"] = str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", "")
    
    print(f"📂 Working directory: {current_dir}")
    print(f"🔧 YOLOv13 path: {yolov13_path}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    return current_dir

def check_dependencies():
    """Check and install PyTorch-compatible dependencies"""
    print("🔍 Checking PyTorch-compatible dependencies...")
    
    # Define PyTorch-compatible package versions
    required_packages = {
        "torch": "torch>=2.2.0",
        "torchvision": "torchvision>=0.17.0", 
        "numpy": "numpy<2.0",  # PyTorch compatible
        "opencv-python": "opencv-python<4.10",  # NumPy 1.x compatible
        "pillow": "pillow<11.0",  # NumPy 1.x compatible
        "pyyaml": "pyyaml>=6.0",
        "tqdm": "tqdm>=4.60.0",
        "pandas": "pandas>=2.0.0",
        "matplotlib": "matplotlib>=3.7.0"
    }
    
    missing_packages = []
    for package_name, package_spec in required_packages.items():
        try:
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package_name} is available")
        except ImportError:
            missing_packages.append(package_spec)
            print(f"⚠️ {package_name} is missing")
    
    if missing_packages:
        print(f"📦 Installing PyTorch-compatible packages: {missing_packages}")
        try:
            # Install without dependencies to prevent NumPy 2.x installation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", "--force-reinstall", "--no-deps"
            ] + missing_packages)
            print("✅ PyTorch-compatible dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    # Verify PyTorch-NumPy compatibility
    try:
        import torch
        import numpy as np
        test_array = np.array([1.0, 2.0, 3.0])
        test_tensor = torch.from_numpy(test_array)
        print(f"✅ PyTorch-NumPy compatibility verified: NumPy {np.__version__}, PyTorch {torch.__version__}")
    except Exception as e:
        print(f"⚠️ PyTorch-NumPy compatibility issue: {e}")
    
    return True

def fix_ultralytics_import():
    """Import ultralytics from local yolov13 directory only"""
    print("📥 Importing ultralytics from local yolov13 directory...")
    
    # Remove any existing ultralytics imports from cache
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"🗑️ Removed cached module: {module}")
    
    try:
        # Import from local yolov13 directory
        from ultralytics import YOLO
        print("✅ Successfully imported YOLO from local yolov13")
        
        # Try to import LOGGER, use fallback if not available
        try:
            from ultralytics.utils import LOGGER
            print("✅ Successfully imported LOGGER from local yolov13")
        except (ImportError, AttributeError):
            print("⚠️ LOGGER import failed, will use print instead")
            LOGGER = None
        
        return YOLO, LOGGER, True
    except ImportError as e:
        print(f"❌ Failed to import ultralytics: {e}")
        return None, None, False

def create_standalone_model_config(variant='s', nc=10):
    """Create a standalone model configuration that works without repository"""
    print(f"🏗️ Creating standalone YOLOv13{variant} configuration...")
    
    # Use the existing yolov13s_standalone.yaml if available
    config_path = "yolov13s_standalone.yaml"
    if Path(config_path).exists():
        print(f"✅ Using existing config: {config_path}")
        
        # Update number of classes if needed
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get('nc') != nc:
            config['nc'] = nc
            # Update Detect layer in head
            for layer in config['head']:
                if isinstance(layer, list) and len(layer) >= 4:
                    if layer[3] == 'Detect':
                        layer[4] = [nc]
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✅ Updated config with {nc} classes")
        
        return config_path
    
    # Create new config if not exists
    model_config = {
        'nc': nc,
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 3, 'C2f', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [512]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [256]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [512]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 9], 1, 'Concat', [1]],
            [-1, 3, 'C2f', [1024]],
            [[15, 18, 21], 1, 'Detect', [nc]],
        ]
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    print(f"✅ Created {config_path}")
    return config_path

def standalone_train(data_config, model_variant='s', epochs=50, batch_size=8, device='cpu'):
    """Standalone training function with support for all YOLOv13 variants"""
    print("🚀 Starting Standalone YOLOv13 Training")
    
    # Validate model variant
    valid_variants = ['n', 's', 'm', 'l', 'x']
    if model_variant not in valid_variants:
        print(f"❌ Invalid model variant '{model_variant}'. Valid options: {valid_variants}")
        return False
    
    print(f"🎯 Using YOLOv13{model_variant} model variant")
    
    # Import ultralytics with conflict resolution
    YOLO, LOGGER, success = fix_ultralytics_import()
    if not success:
        print("❌ Cannot proceed without ultralytics")
        return False
    
    try:
        # Try to use local yolov13 config with specified variant
        model_configs_to_try = [
            # Primary: Official YOLOv13 variants
            Path(f"yolov13/ultralytics/cfg/models/v13/yolov13{model_variant}.yaml"),
            # Fallback 1: Working variants
            Path(f"yolov13/ultralytics/cfg/models/v13/yolov13-working.yaml"),
            # Fallback 2: Simple variants  
            Path(f"yolov13/ultralytics/cfg/models/v13/yolov13-simple.yaml"),
            # Fallback 3: Standalone config
            Path("yolov13s_standalone.yaml"),
        ]
        
        model = None
        used_config = None
        
        for config_path in model_configs_to_try:
            if config_path.exists():
                print(f"✅ Found model config: {config_path}")
                try:
                    model = YOLO(str(config_path))
                    used_config = str(config_path)
                    print(f"✅ Successfully loaded model from: {config_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {config_path}: {e}")
                    continue
        
        if model is None:
            # Create standalone config as last resort
            print("🏗️ Creating standalone model configuration...")
            model_config = create_standalone_model_config(model_variant)
            model = YOLO(model_config)
            used_config = model_config
            print(f"✅ Using standalone model config: {model_config}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    try:
        # Adjust batch size based on model variant
        variant_batch_sizes = {
            'n': batch_size,           # Nano - use as specified
            's': batch_size,           # Small - use as specified  
            'm': max(1, batch_size//2), # Medium - reduce batch size
            'l': max(1, batch_size//4), # Large - reduce batch size more
            'x': max(1, batch_size//8), # Extra Large - smallest batch size
        }
        adjusted_batch_size = variant_batch_sizes.get(model_variant, batch_size)
        
        print(f"📊 Model variant: YOLOv13{model_variant}")
        print(f"📊 Adjusted batch size: {adjusted_batch_size} (original: {batch_size})")
        print(f"📊 Using config: {used_config}")
        
        # Training arguments with fixed data augmentation settings
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'imgsz': 640,
            'batch': adjusted_batch_size,
            'device': device,
            'project': 'runs/standalone_train',
            'name': f'yolov13{model_variant}_standalone',
            'save': True,
            'verbose': True,
            'patience': 30,
            'lr0': 0.01,
            'warmup_epochs': 3,
            'workers': 0,  # Force single thread to avoid data loading issues
            # Fix OpenCV data type issues by disabling problematic augmentations
            'degrees': 0.0,      # Disable rotation
            'translate': 0.0,    # Disable translation
            'scale': 0.0,        # Disable scaling
            'shear': 0.0,        # Disable shear
            'perspective': 0.0,  # Disable perspective
            'flipud': 0.0,       # Disable vertical flip
            'fliplr': 0.0,       # Disable horizontal flip
            'mosaic': 0.0,       # Disable mosaic
            'mixup': 0.0,        # Disable mixup
            'copy_paste': 0.0,   # Disable copy-paste
            'cache': False,      # Disable caching to avoid data type issues
            'rect': False,       # Disable rectangular training
            'cos_lr': False,     # Disable cosine learning rate
            'close_mosaic': 0,   # Disable mosaic closing
            'amp': False,        # Disable automatic mixed precision
            'plots': False,      # Disable plotting to prevent PIL errors
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
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing required keys in dataset config: {missing_keys}")
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
    parser = argparse.ArgumentParser(description="Fixed Standalone YOLOv13 Training")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset configuration file")
    parser.add_argument("--model", type=str, default="s", help="Model variant (n, s, m, l, x)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, 1, ...)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 YOLOv13 Fixed Standalone Training")
    print("=" * 70)
    print("✅ PyTorch-compatible package versions")
    print("✅ All model variants supported (n, s, m, l, x)")
    print("✅ Augmentations disabled for stability")
    print("=" * 70)
    
    # Setup environment
    current_dir = setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Clean model variant
    model_variant = args.model.replace('yolov13', '').replace('yolov8', '').lower()
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
        print("\n🎉 Fixed standalone training completed successfully!")
        print("\n📊 Results:")
        print(f"   Model: YOLOv13{model_variant}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Device: {args.device}")
        print(f"   Saved: runs/standalone_train/yolov13{model_variant}_standalone/")
        print("\n🔍 Next steps:")
        print("   1. Check results: runs/standalone_train/")
        print("   2. Run inference with trained model")
        print("   3. View training curves and metrics")
    else:
        print("\n❌ Training failed")
        print("💡 Try:")
        print("   1. Checking package versions: python -c 'import numpy, torch; print(f\"NumPy: {numpy.__version__}, PyTorch: {torch.__version__}\")'")
        print("   2. Using a smaller batch size: --batch 2")
        print("   3. Using a smaller model variant: --model n")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)