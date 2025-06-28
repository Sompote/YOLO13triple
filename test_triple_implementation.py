#!/usr/bin/env python3
"""
Test script for YOLOv13 triple input implementation
This script tests the basic functionality of the triple input architecture.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

def test_triple_input_conv():
    """Test the TripleInputConv module."""
    print("Testing TripleInputConv module...")
    
    try:
        from ultralytics.nn.modules.conv import TripleInputConv
        
        # Create test inputs
        batch_size = 2
        channels = 3
        height, width = 640, 640
        
        # Single input test
        single_input = torch.randn(batch_size, channels, height, width)
        
        # Triple input test
        triple_input = [
            torch.randn(batch_size, channels, height, width),  # Primary
            torch.randn(batch_size, channels, height, width),  # Detail1
            torch.randn(batch_size, channels, height, width),  # Detail2
        ]
        
        # Initialize module
        conv_module = TripleInputConv(c1=3, c2=64, k=3, s=1)
        
        print(f"  Input shape (single): {single_input.shape}")
        print(f"  Input shapes (triple): {[x.shape for x in triple_input]}")
        
        # Test single input (fallback)
        output_single = conv_module(single_input)
        print(f"  Output shape (single): {output_single.shape}")
        
        # Test triple input
        output_triple = conv_module(triple_input)
        print(f"  Output shape (triple): {output_triple.shape}")
        
        print("✅ TripleInputConv test passed!")
        return True
        
    except Exception as e:
        print(f"❌ TripleInputConv test failed: {e}")
        return False

def test_model_config():
    """Test loading the triple input model configuration."""
    print("\nTesting model configuration...")
    
    try:
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-triple.yaml"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        # Try to load the YOLO model with the config
        from ultralytics import YOLO
        
        print(f"  Loading model from: {config_path}")
        model = YOLO(str(config_path))
        
        print(f"  Model loaded successfully")
        print(f"  Model type: {type(model.model)}")
        
        # Get model info
        print(f"  Model summary:")
        model.info(verbose=False)
        
        print("✅ Model configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

def test_triple_dataset():
    """Test the triple dataset functionality."""
    print("\nTesting TripleYOLODataset...")
    
    try:
        from ultralytics.data.triple_dataset import TripleYOLODataset
        
        # Create a temporary dataset structure
        test_dir = Path("test_triple_data")
        test_dir.mkdir(exist_ok=True)
        
        dirs = [
            "images/primary",
            "images/detail1", 
            "images/detail2",
            "labels"
        ]
        
        for dir_path in dirs:
            (test_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create dummy images and labels
        import cv2
        
        for i in range(3):
            # Create dummy images
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            cv2.imwrite(str(test_dir / f"images/primary/image_{i}.jpg"), dummy_img)
            cv2.imwrite(str(test_dir / f"images/detail1/image_{i}.jpg"), dummy_img)
            cv2.imwrite(str(test_dir / f"images/detail2/image_{i}.jpg"), dummy_img)
            
            # Create dummy label
            with open(test_dir / f"labels/image_{i}.txt", 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")  # class, x, y, w, h
        
        # Test dataset creation
        dataset = TripleYOLODataset(
            img_path=str(test_dir / "images" / "primary"),
            imgsz=640
        )
        
        print(f"  Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample loaded successfully")
            print(f"  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        print("✅ TripleYOLODataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ TripleYOLODataset test failed: {e}")
        # Cleanup on error
        test_dir = Path("test_triple_data")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
        return False

def test_detect_script():
    """Test the detection script functionality."""
    print("\nTesting detection script...")
    
    try:
        # Create test images
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        import cv2
        
        # Create dummy test images
        for name in ["primary.jpg", "detail1.jpg", "detail2.jpg"]:
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(str(test_dir / name), dummy_img)
        
        print(f"  Test images created in: {test_dir}")
        print(f"  Images: {list(test_dir.glob('*.jpg'))}")
        
        # Import detection functions
        sys.path.insert(0, str(Path(__file__).parent))
        from detect_triple import load_triple_images
        
        # Test image loading
        images = load_triple_images(
            str(test_dir / "primary.jpg"),
            str(test_dir / "detail1.jpg"), 
            str(test_dir / "detail2.jpg")
        )
        
        print(f"  Loaded {len(images)} images")
        print(f"  Image shapes: {[img.shape for img in images]}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        print("✅ Detection script test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Detection script test failed: {e}")
        # Cleanup on error
        test_dir = Path("test_images")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("YOLOv13 Triple Input Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("TripleInputConv Module", test_triple_input_conv),
        ("Model Configuration", test_model_config),
        ("Triple Dataset", test_triple_dataset),
        ("Detection Script", test_detect_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Triple input implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("\nNext steps:")
    print("1. Create sample triple images: python detect_triple.py --create-samples")
    print("2. Test detection: python detect_triple.py --primary sample_data/primary/image_1.jpg --detail1 sample_data/detail1/image_1.jpg --detail2 sample_data/detail2/image_1.jpg")
    print("3. Setup training data: python train_triple.py --setup-dirs")
    print("4. Train model: python train_triple.py --data triple_dataset.yaml")

if __name__ == "__main__":
    main()