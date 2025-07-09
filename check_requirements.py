#!/usr/bin/env python3
"""
YOLOv13 Triple Input Model - Requirements Checker
This script verifies that all required dependencies are installed and compatible
"""

import sys
import os
import importlib
import pkg_resources
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Python Version Check")
    
    version = sys.version_info
    print(f"   Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("   ❌ Python 3.8 or higher is required")
        return False
    elif version >= (3, 12):
        print("   ⚠️  Python 3.12+ detected - some packages may have compatibility issues")
        return True
    else:
        print("   ✅ Python version is compatible")
        return True

def check_system_info():
    """Display system information"""
    print("\n💻 System Information")
    
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python path: {sys.executable}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Running in virtual environment")
    else:
        print("   ⚠️  Not running in virtual environment (recommended)")

def check_core_dependencies():
    """Check core dependencies"""
    print("\n📦 Core Dependencies Check")
    
    core_deps = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
    ]
    
    missing = []
    for module, name in core_deps:
        try:
            importlib.import_module(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Not installed")
            missing.append(name)
    
    return len(missing) == 0

def check_optional_dependencies():
    """Check optional dependencies"""
    print("\n🔧 Optional Dependencies Check")
    
    optional_deps = [
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'Progress bars'),
        ('psutil', 'System monitoring'),
        ('scipy', 'Scientific computing'),
        ('pandas', 'Data manipulation'),
        ('albumentations', 'Image augmentation'),
        ('onnx', 'ONNX runtime'),
        ('timm', 'Model library'),
    ]
    
    available = 0
    for module, name in optional_deps:
        try:
            importlib.import_module(module)
            print(f"   ✅ {name}")
            available += 1
        except ImportError:
            print(f"   ⚠️  {name} - Not installed (optional)")
    
    print(f"   📊 {available}/{len(optional_deps)} optional dependencies available")
    return available

def check_cuda_support():
    """Check CUDA support"""
    print("\n🔥 CUDA Support Check")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print("   ✅ CUDA is available")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
            
        return cuda_available
        
    except ImportError:
        print("   ❌ PyTorch not available - cannot check CUDA")
        return False

def check_model_files():
    """Check for model files and structure"""
    print("\n📁 Model Files Check")
    
    required_files = [
        'yolov13/',
        'training_data_demo/',
        'triple_dataset.yaml',
        'demo_verification.py',
        'train_triple.py',
        'detect_triple.py',
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            missing.append(file_path)
    
    return len(missing) == 0

def check_requirements_files():
    """Check for requirements files"""
    print("\n📋 Requirements Files Check")
    
    req_files = [
        'requirements-deployment.txt',
        'requirements-minimal.txt',
        'requirements.txt',
    ]
    
    found = []
    for req_file in req_files:
        if Path(req_file).exists():
            print(f"   ✅ {req_file}")
            found.append(req_file)
        else:
            print(f"   ⚠️  {req_file} - Not found")
    
    return len(found) > 0

def check_package_versions():
    """Check specific package versions"""
    print("\n🔍 Package Versions Check")
    
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        
        print(f"   PyTorch: {torch.__version__}")
        print(f"   TorchVision: {torchvision.__version__}")
        print(f"   OpenCV: {cv2.__version__}")
        print(f"   NumPy: {np.__version__}")
        
        # Check version compatibility
        torch_version = torch.__version__.split('+')[0]  # Remove CUDA suffix
        torch_major, torch_minor = torch_version.split('.')[:2]
        
        if int(torch_major) >= 2:
            print("   ✅ PyTorch version is recent")
        else:
            print("   ⚠️  PyTorch version is old - consider upgrading")
            
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot check versions: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Quick Functionality Test")
    
    try:
        # Test PyTorch
        import torch
        x = torch.randn(1, 3, 224, 224)
        print("   ✅ PyTorch tensor operations work")
        
        # Test OpenCV
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite('test_image.jpg', img)
        loaded = cv2.imread('test_image.jpg')
        os.remove('test_image.jpg')
        print("   ✅ OpenCV image operations work")
        
        # Test model loading (if available)
        if Path('yolov13/ultralytics/cfg/models/v13/yolov13.yaml').exists():
            print("   ✅ Model configuration files accessible")
        else:
            print("   ⚠️  Model configuration files not found")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False

def generate_report():
    """Generate a comprehensive report"""
    print_header("YOLOv13 Triple Input Model - Requirements Report")
    
    results = {
        'python_version': check_python_version(),
        'core_deps': check_core_dependencies(),
        'optional_deps': check_optional_dependencies(),
        'cuda_support': check_cuda_support(),
        'model_files': check_model_files(),
        'requirements_files': check_requirements_files(),
        'package_versions': check_package_versions(),
        'quick_test': run_quick_test(),
    }
    
    check_system_info()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check.replace('_', ' ').title():.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    # Recommendations
    print("\n📋 Recommendations:")
    
    if not results['python_version']:
        print("   • Upgrade Python to 3.8 or higher")
    
    if not results['core_deps']:
        print("   • Install core dependencies: pip install -r requirements-minimal.txt")
    
    if not results['cuda_support']:
        print("   • Install CUDA-enabled PyTorch for GPU acceleration")
    
    if not results['model_files']:
        print("   • Ensure all model files are present in the project directory")
    
    if not results['requirements_files']:
        print("   • Create requirements files for dependency management")
    
    if passed == total:
        print("\n🎉 All checks passed! The environment is ready for deployment.")
        print("\nNext steps:")
        print("   1. Run: python demo_verification.py")
        print("   2. Test: python detect_triple.py --help")
        print("   3. Train: python train_triple.py --help")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please address the issues above.")
    
    return passed == total

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--quiet':
        # Quiet mode - just return exit code
        result = generate_report()
        sys.exit(0 if result else 1)
    else:
        # Normal mode - full report
        result = generate_report()
        return result

if __name__ == "__main__":
    main()