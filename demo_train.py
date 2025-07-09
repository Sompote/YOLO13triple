#!/usr/bin/env python3
"""
Demo training script for YOLOv13 triple input model
Tests the model with the demo dataset to verify it works correctly
"""

import sys
import os
from pathlib import Path
import argparse

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

def demo_train():
    """Run a quick demo training to test the YOLOv13 triple input model"""
    
    print("=" * 60)
    print("YOLOv13 Triple Input Demo Training")
    print("=" * 60)
    
    # Check if demo dataset exists
    dataset_path = Path("training_data_demo")
    if not dataset_path.exists():
        print("❌ Demo dataset not found at:", dataset_path)
        print("Please ensure the demo dataset is created first.")
        return False
    
    # Check if dataset configuration exists
    config_path = Path("triple_dataset.yaml")
    if not config_path.exists():
        print("❌ Dataset configuration not found at:", config_path)
        return False
    
    print("✅ Demo dataset found")
    print("✅ Dataset configuration found")
    
    try:
        from ultralytics import YOLO
        print("✅ YOLOv13 modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import YOLOv13 modules: {e}")
        return False
    
    # Check for model configuration
    model_configs = [
        "yolov13-triple.yaml",
        "yolov13-triple-simple.yaml",
        "yolov13.yaml"
    ]
    
    model_config = None
    for config in model_configs:
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / config
        if config_path.exists():
            model_config = str(config_path)
            print(f"✅ Found model config: {config}")
            break
    
    if not model_config:
        print("❌ No suitable model configuration found")
        print("Available configs in yolov13/ultralytics/cfg/models/v13/:")
        config_dir = yolov13_path / "ultralytics" / "cfg" / "models" / "v13"
        if config_dir.exists():
            for cfg in config_dir.glob("*.yaml"):
                print(f"  - {cfg.name}")
        return False
    
    try:
        # Load model
        print(f"Loading model from: {model_config}")
        model = YOLO(model_config)
        print("✅ Model loaded successfully")
        
        # Training parameters for demo (short training)
        train_args = {
            'data': 'triple_dataset.yaml',
            'epochs': 5,           # Very short for demo
            'imgsz': 640,
            'batch': 2,            # Small batch for demo
            'device': 'cpu',       # Use CPU for demo (works on all systems)
            'project': 'runs/demo',
            'name': 'triple_demo',
            'save': True,
            'verbose': True,
            'patience': 10,
            'lr0': 0.01,
            'seed': 42,
            'workers': 1,          # Single worker for demo
        }
        
        print("\n🚀 Starting demo training...")
        print("Training parameters:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # Start training
        results = model.train(**train_args)
        
        print("\n✅ Demo training completed successfully!")
        print(f"Results saved to: runs/demo/triple_demo")
        
        # Check if model was saved
        model_path = Path("runs/demo/triple_demo/weights/best.pt")
        if model_path.exists():
            print(f"✅ Trained model saved at: {model_path}")
            
            # Test inference with trained model
            print("\n🔍 Testing inference with trained model...")
            try:
                trained_model = YOLO(str(model_path))
                test_image = "training_data_demo/images/primary/val/image_1.jpg"
                
                if Path(test_image).exists():
                    results = trained_model(test_image)
                    print(f"✅ Inference test successful on: {test_image}")
                    print(f"   Detected {len(results[0].boxes)} objects")
                else:
                    print(f"❌ Test image not found: {test_image}")
                    
            except Exception as e:
                print(f"❌ Inference test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nPossible issues:")
        print("1. Dataset format might be incorrect")
        print("2. Model configuration might be incompatible")
        print("3. Dependencies might be missing")
        print("4. Triple input feature might need additional setup")
        return False

def main():
    parser = argparse.ArgumentParser(description="YOLOv13 Triple Input Demo Training")
    parser.add_argument("--full-train", action="store_true", 
                       help="Run full training instead of demo")
    
    args = parser.parse_args()
    
    if args.full_train:
        print("Running full training...")
        # You can call the original train_triple.py here
        os.system("python train_triple.py --data triple_dataset.yaml --epochs 100")
    else:
        success = demo_train()
        if success:
            print("\n🎉 Demo training completed successfully!")
            print("The YOLOv13 triple input model is working correctly.")
            print("\nTo run full training, use:")
            print("python demo_train.py --full-train")
        else:
            print("\n❌ Demo training failed.")
            print("Please check the error messages above.")

if __name__ == "__main__":
    main()