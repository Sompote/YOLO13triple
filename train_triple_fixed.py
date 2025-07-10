#!/usr/bin/env python3
"""
Fixed training script for YOLOv13 with triple image inputs
This script trains the model to process 3 images where the first contains labels
and the other two provide additional detail information.
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

# Force the local yolov13 ultralytics module to be used
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

# Remove any existing ultralytics import from cache
if 'ultralytics' in sys.modules:
    del sys.modules['ultralytics']
if 'ultralytics.data' in sys.modules:
    del sys.modules['ultralytics.data']

# Now import from the local yolov13 directory
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.data.triple_dataset import TripleYOLODataset
    print("✅ Successfully imported YOLOv13 triple input modules")
    
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

    def train_triple_model(data_config, model_config="yolov13s", 
                          epochs=100, imgsz=640, batch_size=16, 
                          device="auto", project="runs/train", name="triple_yolo"):
        """
        Train YOLOv13 model with triple input configuration.
        """
        # Map model names to actual config files
        model_mapping = {
            'yolov13s': 'yolov13s.yaml',
            'yolov13m': 'yolov13m.yaml', 
            'yolov13l': 'yolov13l.yaml',
            'yolov13x': 'yolov13x.yaml',
            'yolov13n': 'yolov13n.yaml',
            'yolov13': 'yolov13.yaml'
        }
        
        config_file = model_mapping.get(model_config, f"{model_config}.yaml")
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / config_file
        
        if not config_path.exists():
            print(f"❌ Model config file not found: {config_path}")
            print("Available configs:")
            config_dir = config_path.parent
            if config_dir.exists():
                for cfg in config_dir.glob("*.yaml"):
                    print(f"  - {cfg.stem}")
            return None
        
        try:
            # Load YOLOv13 model with triple input config
            print(f"🚀 Loading YOLOv13 model: {config_file}")
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
                'save_period': 10,
                'patience': 50,
                'verbose': True,
                'seed': 42,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.0005,
            }
            
            print("🎯 Starting YOLOv13 triple input training...")
            print(f"📊 Training parameters: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
            print("✅ Training completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            print("Please check your data configuration and model setup.")
            return None

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
        print("=" * 50)
        
        # Check if data file exists
        if not Path(args.data).exists():
            print(f"❌ Data file not found: {args.data}")
            print("Please make sure your dataset YAML file exists.")
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
            print(f"📊 Results saved to: {results}")
        else:
            print("❌ Training failed. Please check the logs above.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Could not import required modules from the YOLOv13 repository.")
    print("This repository might need additional setup or dependencies.")
    print("📋 Fallback mode: Triple input training not available")
    print("Please ensure all dependencies are installed and the model is properly configured.")
    
    # Fallback function
    def main():
        print("❌ Triple input training is not available due to import errors.")
        print("Please run the fix script: ./fix_ultralytics_conflict.sh")

if __name__ == "__main__":
    main() 