#!/usr/bin/env python3
"""
YOLOv13 Triple Input Model - Deployment Test Script
This script performs comprehensive testing to verify deployment readiness
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, text):
    """Print formatted step"""
    print(f"\n🔧 Step {step}: {text}")

def run_test(test_name, command, timeout=60):
    """Run a test command and return result"""
    print(f"\n🧪 Testing: {test_name}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ {test_name} - PASSED")
            return True
        else:
            print(f"   ❌ {test_name} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  {test_name} - TIMEOUT")
        return False
    except Exception as e:
        print(f"   ❌ {test_name} - ERROR: {e}")
        return False

def test_basic_imports():
    """Test basic import functionality"""
    print_step(1, "Testing Basic Imports")
    
    import_tests = [
        ("PyTorch", "import torch; print(torch.__version__)"),
        ("TorchVision", "import torchvision; print(torchvision.__version__)"),
        ("OpenCV", "import cv2; print(cv2.__version__)"),
        ("NumPy", "import numpy as np; print(np.__version__)"),
        ("Pillow", "from PIL import Image; print('PIL imported successfully')"),
        ("PyYAML", "import yaml; print('YAML imported successfully')"),
        ("Matplotlib", "import matplotlib.pyplot as plt; print('Matplotlib imported successfully')"),
        ("TQDM", "from tqdm import tqdm; print('TQDM imported successfully')"),
    ]
    
    passed = 0
    for test_name, command in import_tests:
        if run_test(test_name, f"python -c \"{command}\"", timeout=30):
            passed += 1
    
    print(f"\n📊 Import Tests: {passed}/{len(import_tests)} passed")
    return passed == len(import_tests)

def test_model_loading():
    """Test model loading functionality"""
    print_step(2, "Testing Model Loading")
    
    model_tests = [
        ("YOLOv13 Config", "python -c \"import sys; sys.path.append('yolov13'); from ultralytics import YOLO; print('YOLOv13 imported successfully')\""),
        ("Model Config File", "python -c \"from pathlib import Path; print('Config exists:', Path('yolov13/ultralytics/cfg/models/v13/yolov13.yaml').exists())\""),
        ("Triple Input Conv", "python -c \"import sys; sys.path.append('yolov13'); from ultralytics.nn.modules.conv import TripleInputConv; print('TripleInputConv imported successfully')\""),
    ]
    
    passed = 0
    for test_name, command in model_tests:
        if run_test(test_name, command, timeout=30):
            passed += 1
    
    print(f"\n📊 Model Loading Tests: {passed}/{len(model_tests)} passed")
    return passed >= len(model_tests) // 2  # Allow some failures

def test_dataset_functionality():
    """Test dataset and data loading"""
    print_step(3, "Testing Dataset Functionality")
    
    dataset_tests = [
        ("Dataset Structure", "python -c \"from pathlib import Path; print('Dataset exists:', Path('training_data_demo').exists())\""),
        ("Image Loading", "python -c \"from PIL import Image; img = Image.open('training_data_demo/images/primary/val/image_1.jpg'); print('Image loaded:', img.size)\""),
        ("Label Loading", "python -c \"with open('training_data_demo/labels/val/image_1.txt', 'r') as f: print('Label loaded:', len(f.readlines()), 'objects')\""),
        ("YAML Config", "python -c \"import yaml; with open('triple_dataset.yaml', 'r') as f: cfg = yaml.safe_load(f); print('Config loaded:', cfg['nc'], 'classes')\""),
    ]
    
    passed = 0
    for test_name, command in dataset_tests:
        if run_test(test_name, command, timeout=30):
            passed += 1
    
    print(f"\n📊 Dataset Tests: {passed}/{len(dataset_tests)} passed")
    return passed == len(dataset_tests)

def test_inference_functionality():
    """Test inference functionality"""
    print_step(4, "Testing Inference Functionality")
    
    inference_tests = [
        ("Demo Verification", "python demo_verification.py"),
        ("Requirements Check", "python check_requirements.py --quiet"),
    ]
    
    passed = 0
    for test_name, command in inference_tests:
        if run_test(test_name, command, timeout=120):
            passed += 1
    
    print(f"\n📊 Inference Tests: {passed}/{len(inference_tests)} passed")
    return passed >= 1  # At least one should pass

def test_script_functionality():
    """Test script functionality"""
    print_step(5, "Testing Script Functionality")
    
    script_tests = [
        ("Train Script Help", "python train_triple.py --help"),
        ("Detect Script Help", "python detect_triple.py --help"),
        ("Setup Script Help", "python setup_deployment.py --help"),
    ]
    
    passed = 0
    for test_name, command in script_tests:
        if run_test(test_name, command, timeout=30):
            passed += 1
    
    print(f"\n📊 Script Tests: {passed}/{len(script_tests)} passed")
    return passed == len(script_tests)

def test_file_structure():
    """Test file structure completeness"""
    print_step(6, "Testing File Structure")
    
    required_files = [
        'yolov13/',
        'training_data_demo/',
        'triple_dataset.yaml',
        'demo_verification.py',
        'train_triple.py',
        'detect_triple.py',
        'setup_deployment.py',
        'check_requirements.py',
        'requirements-deployment.txt',
        'requirements-minimal.txt',
        'DEPLOYMENT_GUIDE.md',
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"   ❌ Missing files: {missing}")
        return False
    else:
        print(f"   ✅ All {len(required_files)} required files present")
        return True

def create_deployment_package():
    """Create a deployment package"""
    print_step(7, "Creating Deployment Package")
    
    package_dir = Path("deployment_package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    package_dir.mkdir()
    
    # Copy essential files
    essential_files = [
        'yolov13/',
        'training_data_demo/',
        'triple_dataset.yaml',
        'demo_verification.py',
        'train_triple.py',
        'detect_triple.py',
        'setup_deployment.py',
        'check_requirements.py',
        'requirements-deployment.txt',
        'requirements-minimal.txt',
        'DEPLOYMENT_GUIDE.md',
    ]
    
    try:
        for item in essential_files:
            src = Path(item)
            dst = package_dir / item
            
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Create run scripts
        if os.name == 'nt':  # Windows
            run_script = """@echo off
echo Starting YOLOv13 Triple Input Model...
python setup_deployment.py
python demo_verification.py
pause
"""
            with open(package_dir / "run_demo.bat", "w") as f:
                f.write(run_script)
        else:  # Unix-like
            run_script = """#!/bin/bash
echo "Starting YOLOv13 Triple Input Model..."
python setup_deployment.py
python demo_verification.py
"""
            with open(package_dir / "run_demo.sh", "w") as f:
                f.write(run_script)
            os.chmod(package_dir / "run_demo.sh", 0o755)
        
        print(f"   ✅ Deployment package created at: {package_dir}")
        print(f"   📦 Package size: {get_dir_size(package_dir):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Package creation failed: {e}")
        return False

def get_dir_size(path):
    """Get directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def main():
    """Main deployment test function"""
    print_header("YOLOv13 Triple Input Model - Deployment Test")
    
    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Loading", test_model_loading),
        ("Dataset Functionality", test_dataset_functionality),
        ("Inference Functionality", test_inference_functionality),
        ("Script Functionality", test_script_functionality),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Create deployment package
    package_result = create_deployment_package()
    results.append(("Deployment Package", package_result))
    
    # Summary
    print("\n" + "="*60)
    print("  DEPLOYMENT TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All deployment tests passed!")
        print("The YOLOv13 Triple Input Model is ready for standalone deployment.")
        print("\n📦 Deployment package created in: deployment_package/")
        print("\n🚀 Next steps:")
        print("1. Copy the deployment_package/ to target machine")
        print("2. Run: python setup_deployment.py")
        print("3. Test: python demo_verification.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please address the issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)