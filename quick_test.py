#!/usr/bin/env python3
"""
Quick inference test for triple YOLO models
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Setup environment"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if yolov13_path.exists():
        if str(yolov13_path) not in sys.path:
            sys.path.insert(0, str(yolov13_path))
    
    os.environ.update({
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "NUMPY_EXPERIMENTAL_DTYPE_API": "0"
    })

def quick_test(weights_path):
    """Quick inference test"""
    
    print("🚀 Quick Triple YOLO Test")
    print("=" * 40)
    
    if not os.path.exists(weights_path):
        print(f"❌ Model not found: {weights_path}")
        return False
    
    setup_environment()
    
    try:
        from ultralytics import YOLO
        
        print(f"📦 Loading model: {weights_path}")
        model = YOLO(weights_path)
        
        # Test inference on test images
        test_images_dir = "my_dataset_4/images/primary/test"
        if os.path.exists(test_images_dir):
            print(f"🎯 Running inference on: {test_images_dir}")
            
            results = model.predict(
                test_images_dir,
                save=True,
                conf=0.01,  # Low confidence for small objects
                verbose=True
            )
            
            print(f"\n📊 Results:")
            for i, result in enumerate(results):
                if result.boxes is not None:
                    num_detections = len(result.boxes)
                    print(f"Image {i+1}: {num_detections} detections")
                    
                    if num_detections > 0:
                        confidences = result.boxes.conf.cpu().numpy()
                        print(f"  Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
                else:
                    print(f"Image {i+1}: 0 detections")
            
            print("✅ Inference completed!")
            print("📁 Results saved to runs/predict/")
            return True
        else:
            print(f"❌ Test images directory not found: {test_images_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <weights_path>")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    success = quick_test(weights_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()