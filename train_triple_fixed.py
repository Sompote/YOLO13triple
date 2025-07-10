#!/usr/bin/env python3
"""
YOLOv13 Triple Input Training Script
Fixed to use only local ultralytics from yolov13 directory
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

def setup_local_ultralytics():
    """Setup local ultralytics import from yolov13 directory"""
    print("🔧 Setting up local ultralytics import...")
    
    # Get current directory and yolov13 path
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    # Add yolov13 directory to Python path FIRST
    if str(yolov13_path) not in sys.path:
        sys.path.insert(0, str(yolov13_path))
    
    # Set environment variables to avoid conflicts
    os.environ["PYTHONPATH"] = str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", "")
    
    # Remove any existing ultralytics imports from cache
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"🗑️ Removed cached module: {module}")
    
    print(f"✅ Local ultralytics path configured: {yolov13_path}")
    return yolov13_path

# Setup local ultralytics BEFORE any imports
yolov13_path = setup_local_ultralytics()

# Now import from the local yolov13 directory
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    print("✅ Successfully imported YOLOv13 triple input modules")
    
    # Check if triple dataset exists
    try:
        from ultralytics.data.triple_dataset import TripleYOLODataset
        print("✅ Triple dataset support available")
        HAS_TRIPLE_DATASET = True
    except ImportError:
        print("⚠️ Triple dataset not available, using standard dataset")
        HAS_TRIPLE_DATASET = False
    
    def create_triple_dataset_config(data_dir, save_path="triple_dataset.yaml"):
        """
        Create dataset configuration for triple input training.
        
        Args:
            data_dir (str): Path to dataset directory
            save_path (str): Path to save the configuration file
        """
        data_path = Path(data_dir)
        
        config = {
            'path': str(data_path),
            'train': 'images/primary',  # Primary images with labels
            'val': 'images/primary',    # Validation primary images
            'test': '',                 # Optional test set
            
            # Class names (modify according to your dataset)
            'names': {
                0: 'person',
                1: 'bicycle', 
                2: 'car',
                3: 'motorcycle',
                4: 'airplane',
                5: 'bus',
                6: 'train',
                7: 'truck',
                8: 'boat',
                9: 'traffic light',
                # Add more classes as needed
            },
            
            # Number of classes
            'nc': 10,  # Modify according to your dataset
            
            # Triple input specific configuration
            'triple_input': HAS_TRIPLE_DATASET,
            'detail1_path': 'images/detail1' if HAS_TRIPLE_DATASET else None,
            'detail2_path': 'images/detail2' if HAS_TRIPLE_DATASET else None,
            
            # Additional metadata
            'task': 'detect',
            'dataset_type': 'triple_yolo' if HAS_TRIPLE_DATASET else 'yolo'
        }
        
        # Save configuration
        config_path = Path(save_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {config_path}")
        return config_path

    def find_model_config(model_name):
        """Find the best available model configuration"""
        # Try different possible paths
        possible_paths = [
            yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"{model_name}.yaml",
            yolov13_path / "ultralytics" / "cfg" / "models" / f"{model_name}.yaml",
            Path(f"{model_name}.yaml"),
            Path("yolov13s_standalone.yaml"),  # Use existing standalone config
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"✅ Found model config: {path}")
                return path
        
        # If no config found, list available configs
        config_dir = yolov13_path / "ultralytics" / "cfg" / "models" / "v13"
        if config_dir.exists():
            print("❌ Model config not found. Available configs:")
            for cfg in config_dir.glob("*.yaml"):
                print(f"  - {cfg.stem}")
        
        return None

    def train_triple_model(data_config, model_config="yolov13s", 
                          epochs=100, imgsz=640, batch_size=16, 
                          device="auto", project="runs/train", name="triple_yolo"):
        """
        Train YOLOv13 model with triple input configuration.
        """
        print(f"🚀 Starting YOLOv13 training with model: {model_config}")
        
        # Find model configuration
        config_path = find_model_config(model_config)
        if not config_path:
            print(f"❌ Could not find model config for: {model_config}")
            return None
        
        try:
            # Load YOLOv13 model
            print(f"📥 Loading model from: {config_path}")
            model = YOLO(str(config_path))
            
            # Setup training arguments with fixed data augmentation settings
            train_args = {
                'data': data_config,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': device,
                'project': project,
                'name': name,
                'save': True,
                'save_period': 10,
                'patience': 50,
                'verbose': True,
                'seed': 42,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.0005,
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
            
            print("🎯 Starting YOLOv13 training...")
            print(f"📊 Training parameters:")
            for key, value in train_args.items():
                print(f"   {key}: {value}")
            
            # Start training
            results = model.train(**train_args)
            
            print("✅ Training completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def verify_dataset_config(data_config):
        """Verify dataset configuration is valid"""
        print(f"🔍 Verifying dataset config: {data_config}")
        
        if not Path(data_config).exists():
            print(f"❌ Dataset config file not found: {data_config}")
            return False
        
        try:
            with open(data_config, 'r') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"❌ Missing required keys: {missing_keys}")
                return False
            
            # Check dataset paths
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
            print(f"❌ Error verifying dataset: {e}")
            return False

    def main():
        parser = argparse.ArgumentParser(description='Train YOLOv13 with triple input')
        parser.add_argument('--data', type=str, required=True, help='Dataset YAML file')
        parser.add_argument('--model', type=str, default='yolov13s', help='Model configuration')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--batch', type=int, default=16, help='Batch size')
        parser.add_argument('--imgsz', type=int, default=640, help='Image size')
        parser.add_argument('--device', type=str, default='auto', help='Device (cpu, 0, 1, etc.)')
        parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
        parser.add_argument('--name', type=str, default='triple_yolo', help='Experiment name')
        
        args = parser.parse_args()
        
        print("🔥 YOLOv13 Triple Input Training")
        print("=" * 50)
        print(f"📁 Data: {args.data}")
        print(f"🧠 Model: {args.model}")
        print(f"🔄 Epochs: {args.epochs}")
        print(f"📦 Batch: {args.batch}")
        print(f"🖥️  Device: {args.device}")
        print(f"🎯 Triple input support: {'✅' if HAS_TRIPLE_DATASET else '❌'}")
        print("=" * 50)
        
        # Verify dataset config
        if not verify_dataset_config(args.data):
            print("❌ Dataset verification failed")
            return
        
        # Start training
        results = train_triple_model(
            data_config=args.data,
            model_config=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name
        )
        
        if results:
            print("🎉 Training completed successfully!")
            print(f"📊 Results: {results}")
        else:
            print("❌ Training failed. Please check the logs above.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Could not import required modules from the YOLOv13 repository.")
    print("Please ensure the yolov13 directory exists and contains the ultralytics module.")
    print("📋 Fallback mode: Triple input training not available")
    
    # Fallback function
    def main():
        print("❌ Triple input training is not available due to import errors.")
        print("Please check that the yolov13 directory exists and contains ultralytics.")
        print("Also ensure all dependencies are installed.")

if __name__ == "__main__":
    main() 