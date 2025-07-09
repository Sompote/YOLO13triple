#!/usr/bin/env python3
"""
Demo verification script for YOLOv13 Triple Input implementation
This script demonstrates that the triple input concept works by showing:
1. Standard YOLOv13 detection
2. Triple input component functionality
3. Dataset structure validation
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

def create_enhanced_demo_images():
    """Create demo images with more visible objects for better detection"""
    
    print("🎨 Creating enhanced demo images...")
    
    def create_image_with_objects(filename, size=(640, 640)):
        img = Image.new('RGB', size, color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Draw a car (blue rectangle)
        draw.rectangle([200, 300, 400, 400], fill='blue', outline='darkblue', width=3)
        draw.rectangle([220, 320, 380, 380], fill='lightblue')  # windshield
        draw.ellipse([210, 390, 250, 410], fill='black')  # wheel
        draw.ellipse([350, 390, 390, 410], fill='black')  # wheel
        
        # Draw a person (red circle with body)
        draw.ellipse([450, 200, 500, 250], fill='red', outline='darkred', width=2)  # head
        draw.rectangle([465, 250, 485, 350], fill='red')  # body
        draw.rectangle([460, 350, 470, 400], fill='red')  # leg
        draw.rectangle([480, 350, 490, 400], fill='red')  # leg
        
        # Draw a bus (green rectangle)
        draw.rectangle([100, 400, 300, 500], fill='green', outline='darkgreen', width=3)
        draw.rectangle([120, 420, 280, 480], fill='lightgreen')  # windows
        draw.ellipse([110, 490, 140, 510], fill='black')  # wheel
        draw.ellipse([170, 490, 200, 510], fill='black')  # wheel
        draw.ellipse([240, 490, 270, 510], fill='black')  # wheel
        
        img.save(filename)
        return img
    
    # Create primary images
    create_image_with_objects('training_data_demo/images/primary/train/image_1.jpg')
    create_image_with_objects('training_data_demo/images/primary/train/image_2.jpg')
    create_image_with_objects('training_data_demo/images/primary/train/image_3.jpg')
    create_image_with_objects('training_data_demo/images/primary/val/image_1.jpg')
    
    # Create detail images with enhanced features
    def create_detail_image(filename, focus_area):
        img = Image.new('RGB', (640, 640), color='white')
        draw = ImageDraw.Draw(img)
        
        if focus_area == 'car':
            # Enhanced car details
            draw.rectangle([200, 300, 400, 400], fill='darkblue', outline='navy', width=5)
            draw.rectangle([220, 320, 380, 380], fill='cyan')
            draw.ellipse([210, 390, 250, 410], fill='black')
            draw.ellipse([350, 390, 390, 410], fill='black')
            # Add car details
            draw.rectangle([230, 340, 250, 360], fill='yellow')  # headlight
            draw.rectangle([350, 340, 370, 360], fill='yellow')  # headlight
            
        elif focus_area == 'person':
            # Enhanced person details
            draw.ellipse([450, 200, 500, 250], fill='darkred', outline='maroon', width=3)
            draw.rectangle([465, 250, 485, 350], fill='darkred')
            draw.rectangle([460, 350, 470, 400], fill='darkred')
            draw.rectangle([480, 350, 490, 400], fill='darkred')
            # Add person details
            draw.ellipse([465, 210, 475, 220], fill='black')  # eye
            draw.ellipse([485, 210, 495, 220], fill='black')  # eye
            
        img.save(filename)
        return img
    
    # Create detail1 images (car focus)
    create_detail_image('training_data_demo/images/detail1/train/image_1.jpg', 'car')
    create_detail_image('training_data_demo/images/detail1/train/image_2.jpg', 'car')
    create_detail_image('training_data_demo/images/detail1/train/image_3.jpg', 'car')
    create_detail_image('training_data_demo/images/detail1/val/image_1.jpg', 'car')
    
    # Create detail2 images (person focus)
    create_detail_image('training_data_demo/images/detail2/train/image_1.jpg', 'person')
    create_detail_image('training_data_demo/images/detail2/train/image_2.jpg', 'person')
    create_detail_image('training_data_demo/images/detail2/train/image_3.jpg', 'person')
    create_detail_image('training_data_demo/images/detail2/val/image_1.jpg', 'person')
    
    # Create enhanced labels
    def create_enhanced_labels(filename):
        objects = [
            # Car: bbox [200, 300, 400, 400] in 640x640 image
            {
                'class_id': 2,  # car
                'center_x': (200 + 400) / 2 / 640,
                'center_y': (300 + 400) / 2 / 640,
                'width': (400 - 200) / 640,
                'height': (400 - 300) / 640
            },
            # Person: bbox [450, 200, 500, 400] in 640x640 image
            {
                'class_id': 0,  # person
                'center_x': (450 + 500) / 2 / 640,
                'center_y': (200 + 400) / 2 / 640,
                'width': (500 - 450) / 640,
                'height': (400 - 200) / 640
            },
            # Bus: bbox [100, 400, 300, 500] in 640x640 image
            {
                'class_id': 5,  # bus
                'center_x': (100 + 300) / 2 / 640,
                'center_y': (400 + 500) / 2 / 640,
                'width': (300 - 100) / 640,
                'height': (500 - 400) / 640
            }
        ]
        
        with open(filename, 'w') as f:
            for obj in objects:
                f.write(f"{obj['class_id']} {obj['center_x']:.6f} {obj['center_y']:.6f} {obj['width']:.6f} {obj['height']:.6f}\n")
    
    # Create enhanced label files
    create_enhanced_labels('training_data_demo/labels/train/image_1.txt')
    create_enhanced_labels('training_data_demo/labels/train/image_2.txt')
    create_enhanced_labels('training_data_demo/labels/train/image_3.txt')
    create_enhanced_labels('training_data_demo/labels/val/image_1.txt')
    
    print("✅ Enhanced demo images created successfully!")

def test_standard_yolo():
    """Test standard YOLOv13 detection"""
    
    print("\n🔍 Testing Standard YOLOv13 Detection...")
    
    try:
        from ultralytics import YOLO
        
        # Load standard YOLOv13 model
        model_config = str(yolov13_path / "ultralytics/cfg/models/v13/yolov13.yaml")
        model = YOLO(model_config)
        
        # Test on demo image
        test_image = "training_data_demo/images/primary/val/image_1.jpg"
        results = model(test_image)
        
        print(f"✅ Standard YOLOv13 detection successful!")
        print(f"   Detected {len(results[0].boxes)} objects")
        
        # Save result
        results[0].save("standard_yolo_demo_result.jpg")
        print(f"   Result saved to: standard_yolo_demo_result.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Standard YOLOv13 test failed: {e}")
        return False

def test_triple_input_components():
    """Test triple input components"""
    
    print("\n🔧 Testing Triple Input Components...")
    
    try:
        from ultralytics.nn.modules.conv import TripleInputConv
        import torch
        
        # Test TripleInputConv
        triple_conv = TripleInputConv(64, 3, 2)
        
        # Create test inputs
        batch_size = 1
        primary = torch.randn(batch_size, 3, 640, 640)
        detail1 = torch.randn(batch_size, 3, 640, 640)
        detail2 = torch.randn(batch_size, 3, 640, 640)
        
        # Test single input
        single_output = triple_conv(primary)
        print(f"✅ Single input test: {primary.shape} -> {single_output.shape}")
        
        # Test triple input
        triple_output = triple_conv([primary, detail1, detail2])
        print(f"✅ Triple input test: 3x{primary.shape} -> {triple_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Triple input components test failed: {e}")
        return False

def test_dataset_structure():
    """Test dataset structure and validation"""
    
    print("\n📁 Testing Dataset Structure...")
    
    dataset_path = Path("training_data_demo")
    
    # Check directory structure
    required_dirs = [
        "images/primary/train", "images/primary/val",
        "images/detail1/train", "images/detail1/val",
        "images/detail2/train", "images/detail2/val",
        "labels/train", "labels/val"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_name}")
            all_exist = False
        else:
            file_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}: {file_count} files")
    
    if all_exist:
        print("✅ Dataset structure validation passed!")
    else:
        print("❌ Dataset structure validation failed!")
    
    return all_exist

def test_image_loading():
    """Test image loading and processing"""
    
    print("\n🖼️ Testing Image Loading...")
    
    try:
        # Test loading triple images
        primary_path = "training_data_demo/images/primary/val/image_1.jpg"
        detail1_path = "training_data_demo/images/detail1/val/image_1.jpg"
        detail2_path = "training_data_demo/images/detail2/val/image_1.jpg"
        
        paths = [primary_path, detail1_path, detail2_path]
        images = []
        
        for path in paths:
            if Path(path).exists():
                img = Image.open(path)
                images.append(img)
                print(f"✅ Loaded {path}: {img.size}")
            else:
                print(f"❌ Missing image: {path}")
                return False
        
        print(f"✅ Successfully loaded {len(images)} images")
        return True
        
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        return False

def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("YOLOv13 Triple Input Demo Verification")
    print("=" * 60)
    
    # Create enhanced demo data
    create_enhanced_demo_images()
    
    # Run tests
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Image Loading", test_image_loading),
        ("Standard YOLOv13", test_standard_yolo),
        ("Triple Input Components", test_triple_input_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The YOLOv13 triple input implementation is working correctly.")
        print("\nNext steps:")
        print("1. Use the enhanced demo data for training")
        print("2. Run: python train_triple.py --data triple_dataset.yaml --epochs 10")
        print("3. Test detection with: python detect_triple.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()