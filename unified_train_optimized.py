#!/usr/bin/env python3
"""
Optimized YOLOv13 Training Script
Auto-detects single or triple input mode based on dataset configuration
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
    
    # Clean imports and setup path
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    # Set environment
    os.environ.update({
        "PYTHONPATH": str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", ""),
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "ULTRALYTICS_OFFLINE": "1",
        "NUMPY_EXPERIMENTAL_DTYPE_API": "0"
    })
    
    return yolov13_path

def detect_input_mode(data_config):
    """Detect if dataset is single or triple input"""
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        triple_indicators = [
            config.get('triple_input', False),
            'detail1_path' in config,
            'detail2_path' in config,
            config.get('dataset_type') == 'triple_yolo'
        ]
        
        if any(triple_indicators):
            base_path = Path(config['path'])
            detail1_path = base_path / config.get('detail1_path', 'images/detail1')
            detail2_path = base_path / config.get('detail2_path', 'images/detail2')
            
            return 'triple' if detail1_path.exists() and detail2_path.exists() else 'single'
        
        return 'single'
    except:
        return 'single'

def get_model_config_path(yolov13_path, model_variant, input_mode):
    """Get appropriate model configuration path"""
    model_configs = []
    
    if input_mode == 'triple':
        model_configs.extend([
            f"yolov13{model_variant}_triple.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}_triple.yaml"
        ])
    
    model_configs.extend([
        f"yolov13{model_variant}_standalone.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13{model_variant}.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-working.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-simple.yaml",
        "yolov13s_standalone.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s.yaml",
        yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n.yaml"
    ])
    
    for config in model_configs:
        if Path(config).exists():
            return str(config)
    
    return None

def train_model(data_config, model_variant='s', epochs=50, batch_size=4, device='cpu', input_mode='single'):
    """Unified training function"""
    if model_variant not in ['n', 's', 'm', 'l', 'x']:
        return False
    
    yolov13_path = setup_local_ultralytics()
    if yolov13_path is None:
        return False
    
    try:
        from ultralytics import YOLO
        
        if input_mode == 'triple':
            try:
                from ultralytics.data.triple_dataset import TripleYOLODataset
            except ImportError:
                input_mode = 'single'
        
        model_config_path = get_model_config_path(yolov13_path, model_variant, input_mode)
        if not model_config_path:
            return False
        
        model = YOLO(model_config_path)
        
        # Batch size adjustment
        variant_batch_sizes = {
            'n': batch_size,
            's': batch_size,
            'm': max(1, batch_size//2),
            'l': max(1, batch_size//3),
            'x': max(1, batch_size//4)
        }
        adjusted_batch_size = variant_batch_sizes.get(model_variant, batch_size)
        
        # Optimized training configuration
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'batch': adjusted_batch_size,
            'device': device,
            'imgsz': 640,
            'project': f'runs/unified_train_{input_mode}',
            'name': f'yolo_{model_variant}_{input_mode}',
            'save': True,
            'verbose': False,
            'workers': 0,
            'patience': 30 if input_mode == 'single' else 50,
            
            # Disable augmentations for stability
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
        
        # Additional settings for triple input mode
        if input_mode == 'triple':
            train_args.update({
                'lr0': 0.001, 'weight_decay': 0.0005, 'optimizer': 'AdamW',
                'seed': 42, 'save_period': 10
            })
        
        print(f"Training YOLOv13{model_variant} ({input_mode} mode) for {epochs} epochs...")
        results = model.train(**train_args)
        
        print(f"Training completed! Results: runs/unified_train_{input_mode}/yolo_{model_variant}_{input_mode}/")
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
    parser = argparse.ArgumentParser(description='Optimized YOLOv13 Training')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, 0, 1, etc.)')
    parser.add_argument('--variant', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLOv13 model variant')
    parser.add_argument('--force-mode', type=str, choices=['single', 'triple'], 
                       help='Force specific input mode')
    
    args = parser.parse_args()
    
    if not verify_dataset(args.data):
        print("Dataset verification failed")
        return False
    
    input_mode = args.force_mode if args.force_mode else detect_input_mode(args.data)
    
    success = train_model(
        data_config=args.data,
        model_variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        input_mode=input_mode
    )
    
    if success:
        print(f"Training successful! Check runs/unified_train_{input_mode}/yolo_{args.variant}_{input_mode}/")
    else:
        print("Training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)