#!/usr/bin/env python3
"""
Training script for YOLOv13 with triple image inputs
This script trains the model to process 3 images where the first contains labels
and the other two provide additional detail information.
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.data.triple_dataset import TripleYOLODataset
    
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
            'triple_input': True,
            'detail1_path': 'images/detail1',
            'detail2_path': 'images/detail2',
            
            # Additional metadata
            'task': 'detect',
            'dataset_type': 'triple_yolo'
        }
        
        # Save configuration
        config_path = Path(save_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {config_path}")
        return config_path

    def setup_training_directories(base_dir="training_data"):
        """
        Setup directory structure for triple input training.
        
        Args:
            base_dir (str): Base directory for training data
        """
        base_path = Path(base_dir)
        
        # Create directory structure
        dirs = [
            "images/primary/train",
            "images/primary/val", 
            "images/detail1/train",
            "images/detail1/val",
            "images/detail2/train", 
            "images/detail2/val",
            "labels/train",
            "labels/val"
        ]
        
        for dir_path in dirs:
            (base_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Training directory structure created in: {base_path}")
        print("\nDirectory structure:")
        print("training_data/")
        print("├── images/")
        print("│   ├── primary/")
        print("│   │   ├── train/")
        print("│   │   └── val/")
        print("│   ├── detail1/")
        print("│   │   ├── train/")
        print("│   │   └── val/")
        print("│   └── detail2/")
        print("│       ├── train/")
        print("│       └── val/")
        print("└── labels/")
        print("    ├── train/")
        print("    └── val/")
        
        # Create README with instructions
        readme_content = '''# Triple Input YOLO Training Data

## Directory Structure
- `images/primary/`: Primary images with object labels
- `images/detail1/`: First set of detail images
- `images/detail2/`: Second set of detail images  
- `labels/`: YOLO format label files (corresponding to primary images)

## Data Preparation
1. Place your primary images (with objects to detect) in `images/primary/train/` and `images/primary/val/`
2. Place corresponding detail images in `images/detail1/` and `images/detail2/` directories
3. Place YOLO format labels in `labels/train/` and `labels/val/`

## Label Format
Each label file should have the same name as the primary image but with .txt extension.
Label format: `class_id center_x center_y width height` (normalized 0-1)

## Training
Run: `python train_triple.py --data triple_dataset.yaml --model yolov13-triple.yaml`
'''
        
        with open(base_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        return base_path

    def train_triple_model(data_config, model_config="yolov13-triple", 
                          epochs=100, imgsz=640, batch_size=16, 
                          device="auto", project="runs/train", name="triple_yolo"):
        """
        Train YOLOv13 model with triple input configuration.
        
        Args:
            data_config (str): Path to dataset configuration file
            model_config (str): Model configuration name
            epochs (int): Number of training epochs
            imgsz (int): Image size
            batch_size (int): Batch size
            device (str): Device to use for training
            project (str): Project directory
            name (str): Experiment name
        """
        # Use the YOLOv13 config file - support different variants
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"{model_config}.yaml"
        
        if not config_path.exists():
            print(f"Model config file not found: {config_path}")
            print("Available configs:")
            config_dir = config_path.parent
            available_configs = []
            for cfg in config_dir.glob("*.yaml"):
                available_configs.append(cfg.stem)
                print(f"  - {cfg.stem}")
            
            # Try to suggest closest match
            if model_config in ['yolov13s', 'yolov13m', 'yolov13l', 'yolov13x', 'yolov13n']:
                if 'yolov13' in available_configs:
                    print(f"\nSuggestion: Try using 'yolov13' (base model) instead of '{model_config}'")
            return None
        
        try:
            # Load YOLOv13 model with triple input config
            print(f"Loading YOLOv13 triple model from {config_path}...")
            model = YOLO(str(config_path))
            
            # Setup training arguments
            train_args = {
                'data': data_config,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': device,
                'project': project,
                'name': name,
                'save': True,
                'save_period': 10,  # Save checkpoint every 10 epochs
                'patience': 50,     # Early stopping patience
                'verbose': True,
                'seed': 42,        # For reproducibility
                
                # Optimization settings
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                
                # Data augmentation
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.9,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.15,
                'copy_paste': 0.3,
            }
            
            print(f"Starting training with the following configuration:")
            print(f"  Model: {model_config}")
            print(f"  Data: {data_config}")
            print(f"  Epochs: {epochs}")
            print(f"  Image size: {imgsz}")
            print(f"  Batch size: {batch_size}")
            print(f"  Device: {device}")
            
            # Start training
            results = model.train(**train_args)
            
            print("Training completed!")
            print(f"Results saved to: {project}/{name}")
            
            return results
            
        except Exception as e:
            print(f"Error during training: {e}")
            print("Possible issues:")
            print("1. Dataset configuration is incorrect")
            print("2. Triple input modules need proper integration")
            print("3. Insufficient GPU memory (try reducing batch size)")
            print("4. Missing dependencies")
            return None

    def validate_dataset(data_dir):
        """
        Validate triple input dataset structure and files.
        
        Args:
            data_dir (str): Path to dataset directory
        """
        data_path = Path(data_dir)
        
        print(f"Validating dataset at: {data_path}")
        
        # Check directory structure
        required_dirs = [
            "images/primary/train",
            "images/primary/val",
            "images/detail1/train", 
            "images/detail1/val",
            "images/detail2/train",
            "images/detail2/val",
            "labels/train",
            "labels/val"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (data_path / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print("❌ Missing directories:")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
            return False
        
        print("✅ All required directories exist")
        
        # Check for images and labels
        primary_train = list((data_path / "images/primary/train").glob("*"))
        primary_val = list((data_path / "images/primary/val").glob("*"))
        labels_train = list((data_path / "labels/train").glob("*.txt"))
        labels_val = list((data_path / "labels/val").glob("*.txt"))
        
        print(f"📊 Dataset statistics:")
        print(f"  Training images: {len(primary_train)}")
        print(f"  Validation images: {len(primary_val)}")
        print(f"  Training labels: {len(labels_train)}")
        print(f"  Validation labels: {len(labels_val)}")
        
        # Check if we have corresponding detail images
        detail1_train = list((data_path / "images/detail1/train").glob("*"))
        detail2_train = list((data_path / "images/detail2/train").glob("*"))
        
        print(f"  Detail1 training images: {len(detail1_train)}")
        print(f"  Detail2 training images: {len(detail2_train)}")
        
        if len(primary_train) == 0:
            print("❌ No training images found")
            return False
        
        if len(labels_train) == 0:
            print("❌ No training labels found")
            return False
        
        print("✅ Dataset validation passed")
        return True

    def main():
        parser = argparse.ArgumentParser(description="YOLOv13 Triple Input Training")
        parser.add_argument("--data", type=str, help="Path to dataset configuration file")
        parser.add_argument("--model", type=str, default="yolov13-triple", 
                          help="Model configuration")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument("--imgsz", type=int, default=640, help="Image size")
        parser.add_argument("--batch", type=int, default=16, help="Batch size")
        parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, 0, 1, ...)")
        parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
        parser.add_argument("--name", type=str, default="triple_yolo", help="Experiment name")
        
        parser.add_argument("--setup-dirs", action="store_true", 
                          help="Setup training directory structure")
        parser.add_argument("--data-dir", type=str, default="training_data",
                          help="Base directory for training data")
        parser.add_argument("--validate", action="store_true",
                          help="Validate dataset structure")
        parser.add_argument("--create-config", action="store_true",
                          help="Create dataset configuration file")
        
        args = parser.parse_args()
        
        # Setup directories if requested
        if args.setup_dirs:
            setup_training_directories(args.data_dir)
            return
        
        # Validate dataset if requested
        if args.validate:
            if not args.data_dir:
                parser.error("--data-dir is required for validation")
            validate_dataset(args.data_dir)
            return
        
        # Create config if requested
        if args.create_config:
            if not args.data_dir:
                parser.error("--data-dir is required for config creation")
            create_triple_dataset_config(args.data_dir)
            return
        
        # Check if data config is provided
        if not args.data:
            parser.error("--data is required for training, or use utility flags like --setup-dirs")
        
        # Check if data config exists
        if not Path(args.data).exists():
            print(f"Error: Dataset configuration file not found: {args.data}")
            print("\nTo create a configuration file, use:")
            print(f"python {__file__} --create-config --data-dir /path/to/your/dataset")
            return
        
        # Start training
        print("Starting YOLOv13 Triple Input Training...")
        results = train_triple_model(
            data_config=args.data,
            model_config=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            project=args.project,
            name=args.name
        )
        
        if results:
            print("Training completed successfully!")
        else:
            print("Training failed. Please check the error messages above.")

except ImportError as e:
    print(f"Import error: {e}")
    print("Could not import required modules from the YOLOv13 repository.")
    print("This repository might need additional setup or dependencies.")
    
    def main():
        print("Fallback mode: Triple input training not available")
        print("Please ensure all dependencies are installed and the model is properly configured.")

if __name__ == "__main__":
    main()