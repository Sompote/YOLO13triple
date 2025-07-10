#!/usr/bin/env python3
"""
Simple YOLOv13 Training Script
Minimal configuration for reliable training without complex augmentations
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

def setup_local_ultralytics():
    """Setup local ultralytics import with package upgrade protection"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        print(f"❌ YOLOv13 directory not found: {yolov13_path}")
        return None
    
    # Remove any existing ultralytics imports first
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"🗑️ Removed cached module: {module}")
    
    # Add to Python path at the very beginning
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    # Clean environment and prevent package upgrades
    os.environ["PYTHONPATH"] = str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", "")
    
    # Disable automatic updates and package installations
    os.environ["ULTRALYTICS_AUTO_UPDATE"] = "0"
    os.environ["ULTRALYTICS_DISABLE_CHECKS"] = "1" 
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # Force NumPy 1.x compatibility mode
    os.environ["NUMPY_EXPERIMENTAL_DTYPE_API"] = "0"
    
    # Verify the local ultralytics can be found
    ultralytics_init = yolov13_path / "ultralytics" / "__init__.py"
    ultralytics_data_init = yolov13_path / "ultralytics" / "data" / "__init__.py"
    
    if not ultralytics_init.exists():
        print(f"❌ Local ultralytics __init__.py not found: {ultralytics_init}")
        return None
        
    if not ultralytics_data_init.exists():
        print(f"❌ Local ultralytics.data __init__.py not found: {ultralytics_data_init}")
        return None
    
    print(f"✅ Local ultralytics path configured: {yolov13_path}")
    print(f"✅ Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    return yolov13_path

def simple_train(data_config, model_variant='s', epochs=50, batch_size=4, device='cpu'):
    """Simple training function with minimal configuration and support for all variants"""
    print("🚀 Starting Simple YOLOv13 Training")
    
    # Validate model variant
    valid_variants = ['n', 's', 'm', 'l', 'x']
    if model_variant not in valid_variants:
        print(f"❌ Invalid model variant '{model_variant}'. Valid options: {valid_variants}")
        return False
    
    print(f"🎯 Using YOLOv13{model_variant} model variant")
    
    # Setup local ultralytics
    yolov13_path = setup_local_ultralytics()
    
    if yolov13_path is None:
        print("❌ Failed to setup local ultralytics")
        return False
    
    try:
        # Test import step by step
        print("📥 Testing ultralytics import...")
        import ultralytics
        print(f"✅ Ultralytics imported from: {ultralytics.__file__}")
        
        print("📥 Testing YOLO import...")
        from ultralytics import YOLO
        print("✅ Successfully imported YOLO")
        
        # Test if it's our local version
        if str(yolov13_path) in ultralytics.__file__:
            print("✅ Using local YOLOv13 ultralytics implementation")
        else:
            print(f"⚠️ Warning: Using external ultralytics from {ultralytics.__file__}")
            
    except ImportError as e:
        print(f"❌ Failed to import YOLO: {e}")
        print("🔍 Checking what's in the local ultralytics...")
        ultralytics_dir = yolov13_path / "ultralytics"
        if ultralytics_dir.exists():
            print(f"📁 Contents of {ultralytics_dir}:")
            for item in ultralytics_dir.iterdir():
                print(f"   - {item.name}")
        return False
    
    try:
        # Try to use model configs in order of preference
        model_configs_to_try = [
            # Primary: Official YOLOv13 variants
            f"yolov13{model_variant}_standalone.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}.yaml",
            # Fallback 1: Working variants
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-working.yaml",
            # Fallback 2: Simple variants
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-simple.yaml",
            # Fallback 3: Default standalone config
            "yolov13s_standalone.yaml",
            # Fallback 4: Any available variant
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n.yaml",
        ]
        
        model = None
        used_config = None
        
        for config in model_configs_to_try:
            if Path(config).exists():
                print(f"✅ Found model config: {config}")
                try:
                    model = YOLO(str(config))
                    used_config = str(config)
                    print(f"✅ Successfully loaded model from: {config}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {config}: {e}")
                    continue
        
        if model is None:
            print("❌ No suitable model configuration found")
            return False
        
        # Adjust batch size based on model variant for stability
        variant_batch_sizes = {
            'n': batch_size,           # Nano - use as specified
            's': batch_size,           # Small - use as specified
            'm': max(1, batch_size//2), # Medium - reduce batch size
            'l': max(1, batch_size//3), # Large - reduce batch size more
            'x': max(1, batch_size//4), # Extra Large - smallest batch size
        }
        adjusted_batch_size = variant_batch_sizes.get(model_variant, batch_size)
        
        print(f"📊 Model variant: YOLOv13{model_variant}")
        print(f"📊 Adjusted batch size: {adjusted_batch_size} (original: {batch_size})")
        print(f"📊 Using config: {used_config}")
        
        # Minimal training configuration - no augmentations
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'batch': adjusted_batch_size,
            'device': device,
            'imgsz': 640,
            'project': 'runs/simple_train',
            'name': f'simple_yolo_{model_variant}',
            'save': True,
            'verbose': True,
            'workers': 0,
            'patience': 30,
            # Disable ALL augmentations to prevent OpenCV issues
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'cache': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 0,
            'amp': False,
            'single_cls': False,
            'overlap_mask': False,
            'mask_ratio': 1,
            'dropout': 0.0,
            'val': True,
            'plots': False,        # Disable plotting to prevent PIL errors
            'save_json': False,
            'save_hybrid': False,
            'half': False,
            'dnn': False,
        }
        
        print("🎯 Simple Training Configuration:")
        print(f"   Data: {data_config}")
        print(f"   Model: YOLOv13{model_variant}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {adjusted_batch_size}")
        print(f"   Device: {device}")
        print(f"   Image size: 640")
        print(f"   Augmentations: Disabled (for stability)")
        
        print("\n🚀 Starting training...")
        results = model.train(**train_args)
        
        print("✅ Training completed successfully!")
        print(f"💾 Results saved to: runs/simple_train/simple_yolo_{model_variant}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dataset(data_config):
    """Verify dataset configuration"""
    if not Path(data_config).exists():
        print(f"❌ Dataset config not found: {data_config}")
        return False
    
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        print(f"✅ Dataset verified: {config['nc']} classes")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False

def verify_package_compatibility():
    """Verify PyTorch-compatible package versions"""
    print("🔍 Verifying PyTorch-compatible package versions...")
    
    try:
        import numpy as np
        import torch
        import cv2
        import PIL
        
        # Check versions
        numpy_version = np.__version__
        torch_version = torch.__version__
        cv2_version = cv2.__version__
        pil_version = PIL.__version__
        
        print(f"📦 NumPy: {numpy_version}")
        print(f"📦 PyTorch: {torch_version}")
        print(f"📦 OpenCV: {cv2_version}")
        print(f"📦 Pillow: {pil_version}")
        
        # Check NumPy compatibility
        if numpy_version.startswith('2.'):
            print("❌ NumPy 2.x detected - NOT compatible with PyTorch!")
            print("🔧 Installing PyTorch-compatible NumPy...")
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "numpy<2.0", "--force-reinstall", "--no-deps"
            ])
            print("✅ NumPy downgraded to 1.x")
            # Restart Python to reload NumPy
            print("⚠️ Please restart the script to use the new NumPy version")
            return False
        
        # Test PyTorch-NumPy compatibility
        test_array = np.array([1.0, 2.0, 3.0])
        test_tensor = torch.from_numpy(test_array)
        
        print("✅ PyTorch-NumPy compatibility verified")
        return True
        
    except Exception as e:
        print(f"❌ Package compatibility check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple YOLOv13 Training with All Model Variants')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, 0, 1, etc.)')
    parser.add_argument('--variant', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv13 model variant: n(nano), s(small), m(medium), l(large), x(extra-large)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 Simple YOLOv13 Training")
    print("=" * 70)
    print("This script uses minimal configuration for maximum reliability")
    print("All augmentations are disabled to prevent data type issues")
    print("✅ PyTorch-compatible package versions")
    print(f"✅ Model variant: YOLOv13{args.variant}")
    print("=" * 70)
    
    # Verify dataset
    if not verify_dataset(args.data):
        print("❌ Dataset verification failed")
        return False
    
    # Verify package compatibility
    if not verify_package_compatibility():
        print("❌ Package compatibility check failed. Exiting.")
        return False

    # Start training
    success = simple_train(
        data_config=args.data,
        model_variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    if success:
        print("\n🎉 Simple training completed successfully!")
        print("📊 Check results in: runs/simple_train/simple_yolo_<variant>/")
        print(f"💾 Best model: runs/simple_train/simple_yolo_{args.variant}/weights/best.pt")
    else:
        print("\n❌ Training failed")
        print("💡 Try:")
        print("   1. Using a smaller batch size: --batch 2")
        print("   2. Using a smaller model variant: --variant n")
        print("   3. Checking package compatibility")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 