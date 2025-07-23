#!/usr/bin/env python3
"""
Unified YOLOv13 Training Script
Auto-detects single or triple input mode based on dataset configuration
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
    print(f"✅ Python path: {sys.path[:3]}...")
    
    return yolov13_path

def detect_input_mode(data_config):
    """Detect if dataset is single or triple input based on configuration"""
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for triple input indicators
        triple_indicators = [
            config.get('triple_input', False),
            'detail1_path' in config,
            'detail2_path' in config,
            config.get('dataset_type') == 'triple_yolo'
        ]
        
        is_triple = any(triple_indicators)
        
        if is_triple:
            print("🔍 Detected: Triple input dataset")
            # Verify triple input paths exist
            if 'path' in config:
                base_path = Path(config['path'])
                detail1_path = base_path / config.get('detail1_path', 'images/detail1')
                detail2_path = base_path / config.get('detail2_path', 'images/detail2')
                
                if not detail1_path.exists() or not detail2_path.exists():
                    print("⚠️ Triple input paths not found, falling back to single input mode")
                    return 'single'
            return 'triple'
        else:
            print("🔍 Detected: Single input dataset")
            return 'single'
            
    except Exception as e:
        print(f"❌ Error detecting input mode: {e}")
        print("🔄 Defaulting to single input mode")
        return 'single'

def get_model_config_path(yolov13_path, model_variant, input_mode):
    """Get appropriate model configuration path"""
    model_configs_to_try = []
    
    if input_mode == 'triple':
        # Triple input model configurations
        model_configs_to_try.extend([
            f"yolov13{model_variant}_triple.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}_triple.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}.yaml",
        ])
    
    # Common configurations for both modes
    model_configs_to_try.extend([
        f"yolov13{model_variant}_standalone.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-working.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-simple.yaml",
        "yolov13s_standalone.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n.yaml",
    ])
    
    for config in model_configs_to_try:
        if Path(config).exists():
            print(f"✅ Found model config: {config}")
            return str(config)
    
    print("❌ No suitable model configuration found")
    return None

def train_model(data_config, model_variant='s', epochs=50, batch_size=4, device='cpu', input_mode='single'):
    """Unified training function for both single and triple input modes"""
    print(f"🚀 Starting YOLOv13 Training ({input_mode} input mode)")
    
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
        # Import YOLO
        print("📥 Testing ultralytics import...")
        import ultralytics
        print(f"✅ Ultralytics imported from: {ultralytics.__file__}")
        
        from ultralytics import YOLO
        print("✅ Successfully imported YOLO")
        
        # Check for triple dataset support if needed
        if input_mode == 'triple':
            try:
                from ultralytics.data.triple_dataset import TripleYOLODataset
                print("✅ Triple dataset support available")
            except ImportError:
                print("⚠️ Triple dataset not available, using standard training")
                input_mode = 'single'
                
    except ImportError as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False
    
    try:
        # Get model configuration
        model_config_path = get_model_config_path(yolov13_path, model_variant, input_mode)
        if not model_config_path:
            return False
        
        # Load model
        model = YOLO(model_config_path)
        print(f"✅ Successfully loaded model from: {model_config_path}")
        
        # Adjust batch size based on model variant for stability
        variant_batch_sizes = {
            'n': batch_size,
            's': batch_size,
            'm': max(1, batch_size//2),
            'l': max(1, batch_size//3),
            'x': max(1, batch_size//4),
        }
        adjusted_batch_size = variant_batch_sizes.get(model_variant, batch_size)
        
        # Training configuration optimized for both modes
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'batch': adjusted_batch_size,
            'device': device,
            'imgsz': 640,
            'project': f'runs/unified_train_{input_mode}',
            'name': f'yolo_{model_variant}_{input_mode}',
            'save': True,
            'verbose': True,
            'workers': 0,
            'patience': 30 if input_mode == 'single' else 50,
            # Disable problematic augmentations for stability
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
            'plots': False,
            'save_json': False,
            'save_hybrid': False,
            'half': False,
            'dnn': False,
            'conf': 0.01,         # Optimized for small objects
            'iou': 0.3,           # Optimized for small objects
            'max_det': 300,
        }
        
        # Additional settings for triple input mode
        if input_mode == 'triple':
            train_args.update({
                'lr0': 0.001,
                'weight_decay': 0.0005,
                'optimizer': 'AdamW',
                'seed': 42,
                'save_period': 10,
            })
        
        print(f"🎯 Unified Training Configuration:")
        print(f"   Input Mode: {input_mode}")
        print(f"   Data: {data_config}")
        print(f"   Model: YOLOv13{model_variant}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {adjusted_batch_size}")
        print(f"   Device: {device}")
        print(f"   Image size: 640")
        print(f"   Config: {model_config_path}")
        
        print(f"\n🚀 Starting {input_mode} input training...")
        results = model.train(**train_args)
        
        print("✅ Training completed successfully!")
        print(f"💾 Results saved to: runs/unified_train_{input_mode}/yolo_{model_variant}_{input_mode}/")
        
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
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "numpy<2.0", "--force-reinstall", "--no-deps"
            ])
            print("✅ NumPy downgraded to 1.x")
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
    parser = argparse.ArgumentParser(description='Unified YOLOv13 Training (Auto-detects Single/Triple Input)')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, 0, 1, etc.)')
    parser.add_argument('--variant', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv13 model variant: n(nano), s(small), m(medium), l(large), x(extra-large)')
    parser.add_argument('--force-mode', type=str, choices=['single', 'triple'], 
                       help='Force specific input mode (overrides auto-detection)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 Unified YOLOv13 Training")
    print("=" * 70)
    print("Auto-detects single or triple input mode from dataset configuration")
    print("Uses minimal augmentations for maximum stability")
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

    # Detect input mode (unless forced)
    if args.force_mode:
        input_mode = args.force_mode
        print(f"🔧 Forced input mode: {input_mode}")
    else:
        input_mode = detect_input_mode(args.data)
    
    # Start training
    success = train_model(
        data_config=args.data,
        model_variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        input_mode=input_mode
    )
    
    if success:
        print(f"\n🎉 Unified training completed successfully!")
        print(f"📊 Check results in: runs/unified_train_{input_mode}/yolo_{args.variant}_{input_mode}/")
        print(f"💾 Best model: runs/unified_train_{input_mode}/yolo_{args.variant}_{input_mode}/weights/best.pt")
    else:
        print("\n❌ Training failed")
        print("💡 Try:")
        print("   1. Using a smaller batch size: --batch 2")
        print("   2. Using a smaller model variant: --variant n") 
        print("   3. Forcing single input mode: --force-mode single")
        print("   4. Checking package compatibility")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)