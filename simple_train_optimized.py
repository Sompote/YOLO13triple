#!/usr/bin/env python3
"""
Optimized Simple YOLOv13 Training Script
Minimal configuration for reliable training
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

def setup_local_ultralytics():
    """Setup local ultralytics import"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        return None
    
    # Clean imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    # Environment setup
    os.environ.update({
        "PYTHONPATH": str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", ""),
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "ULTRALYTICS_OFFLINE": "1",
        "NUMPY_EXPERIMENTAL_DTYPE_API": "0"
    })
    
    return yolov13_path

def simple_train(data_config, model_variant='s', epochs=50, batch_size=4, device='cpu'):
    """Simple training function"""
    if model_variant not in ['n', 's', 'm', 'l', 'x']:
        return False
    
    yolov13_path = setup_local_ultralytics()
    if yolov13_path is None:
        return False
    
    try:
        from ultralytics import YOLO
        
        # Find model configuration
        model_configs = [
            f"yolov13{model_variant}_standalone.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-working.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-simple.yaml",
            "yolov13s_standalone.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n.yaml"
        ]
        
        model = None
        for config in model_configs:
            if Path(config).exists():
                try:
                    model = YOLO(str(config))
                    break
                except:
                    continue
        
        if model is None:
            return False
        
        # Batch size adjustment
        variant_batch_sizes = {
            'n': batch_size, 's': batch_size, 'm': max(1, batch_size//2),
            'l': max(1, batch_size//3), 'x': max(1, batch_size//4)
        }
        adjusted_batch_size = variant_batch_sizes.get(model_variant, batch_size)
        
        # Minimal training configuration
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'batch': adjusted_batch_size,
            'device': device,
            'imgsz': 640,
            'project': 'runs/simple_train',
            'name': f'simple_yolo_{model_variant}',
            'save': True,
            'verbose': False,
            'workers': 0,
            'patience': 30,
            
            # Disable augmentations
            'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'shear': 0.0,
            'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.0, 'mosaic': 0.0,
            'mixup': 0.0, 'copy_paste': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
            
            # Optimized settings
            'cache': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 0,
            'amp': False, 'single_cls': False, 'overlap_mask': False, 'mask_ratio': 1,
            'dropout': 0.0, 'val': True, 'plots': False, 'save_json': False,
            'save_hybrid': False, 'half': False, 'dnn': False,
            
            # Optimized thresholds for small objects
            'conf': 0.01, 'iou': 0.3, 'max_det': 300
        }
        
        print(f"Training YOLOv13{model_variant} for {epochs} epochs...")
        results = model.train(**train_args)
        
        print(f"Training completed! Results: runs/simple_train/simple_yolo_{model_variant}/")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def verify_dataset(data_config):
    """Verify dataset configuration"""
    if not Path(data_config).exists():
        return False
    
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        required_keys = ['train', 'val', 'nc', 'names']
        return all(key in config for key in required_keys)
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple YOLOv13 Training')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, 0, 1, etc.)')
    parser.add_argument('--variant', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv13 model variant')
    
    args = parser.parse_args()
    
    if not verify_dataset(args.data):
        print("Dataset verification failed")
        return False
    
    success = simple_train(
        data_config=args.data,
        model_variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    if success:
        print(f"Training successful! Check runs/simple_train/simple_yolo_{args.variant}/")
    else:
        print("Training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)