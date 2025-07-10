#!/usr/bin/env python3
"""
Cloud Issues Fixer for YOLOv13 Triple Input
This script diagnoses and fixes common cloud deployment issues.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def print_status(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def check_python_version():
    """Check Python version compatibility"""
    print_info("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print_error("Python 3.8+ required")
        return False
    
    print_status("Python version compatible")
    return True

def fix_numpy_conflict():
    """Fix NumPy version conflicts"""
    print_info("Fixing NumPy version conflicts...")
    
    try:
        import numpy as np
        version = np.__version__
        print(f"Current NumPy version: {version}")
        
        if version.startswith('2.'):
            print_warning("NumPy 2.x detected - downgrading to 1.x for compatibility")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "numpy<2.0", "--force-reinstall", "--no-warn-script-location"
            ])
            print_status("NumPy downgraded successfully")
            
            # Force reimport
            if 'numpy' in sys.modules:
                importlib.reload(sys.modules['numpy'])
        else:
            print_status("NumPy version compatible")
        
        return True
    except Exception as e:
        print_error(f"NumPy fix failed: {e}")
        return False

def clear_python_cache():
    """Clear Python cache to avoid import conflicts"""
    print_info("Clearing Python cache...")
    
    try:
        # Remove __pycache__ directories
        for root, dirs, files in os.walk('.'):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                subprocess.run(['rm', '-rf', pycache_path])
        
        # Clear ultralytics modules from sys.modules
        modules_to_clear = [mod for mod in sys.modules.keys() if 'ultralytics' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        print_status("Python cache cleared")
        return True
    except Exception as e:
        print_error(f"Cache clearing failed: {e}")
        return False

def fix_ultralytics_install():
    """Fix ultralytics installation issues"""
    print_info("Fixing ultralytics installation...")
    
    try:
        # Uninstall and reinstall ultralytics
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", 
            "ultralytics", "-y", "--no-warn-script-location"
        ])
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ultralytics>=8.0.0", "--no-warn-script-location"
        ])
        
        print_status("Ultralytics reinstalled")
        return True
    except Exception as e:
        print_error(f"Ultralytics fix failed: {e}")
        return False

def test_imports():
    """Test critical imports"""
    print_info("Testing critical imports...")
    
    tests = [
        ('numpy', 'import numpy as np'),
        ('torch', 'import torch'),
        ('cv2', 'import cv2'),
        ('yaml', 'import yaml'),
        ('PIL', 'from PIL import Image'),
    ]
    
    failed_imports = []
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print_status(f"{name} import successful")
        except Exception as e:
            print_error(f"{name} import failed: {e}")
            failed_imports.append(name)
    
    # Test ultralytics separately
    try:
        from ultralytics import YOLO
        print_status("ultralytics.YOLO import successful")
    except Exception as e:
        print_warning(f"ultralytics.YOLO import failed: {e}")
        failed_imports.append('ultralytics')
    
    return len(failed_imports) == 0

def create_minimal_test():
    """Create a minimal test script"""
    print_info("Creating minimal test script...")
    
    test_script = '''#!/usr/bin/env python3
"""Minimal YOLOv13 test script"""

import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except Exception as e:
    print("❌ NumPy:", e)

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
except Exception as e:
    print("❌ PyTorch:", e)

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO imported")
    
    # Test model creation
    model = YOLO("yolov8n.yaml")
    print("✅ Model created successfully")
    
except Exception as e:
    print("❌ Ultralytics/Model:", e)

print("\\n🎯 If all imports show ✅, you can run training!")
'''
    
    with open('test_minimal.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_minimal.py', 0o755)
    print_status("Created test_minimal.py")

def main():
    print("=" * 60)
    print("🔧 YOLOv13 Triple Input - Cloud Issues Fixer")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Fix NumPy conflicts
    if not fix_numpy_conflict():
        return False
    
    # Clear cache
    if not clear_python_cache():
        return False
    
    # Fix ultralytics
    if not fix_ultralytics_install():
        return False
    
    # Test imports
    if test_imports():
        print_status("All imports working correctly!")
    else:
        print_warning("Some imports still failing - check individual errors above")
    
    # Create test script
    create_minimal_test()
    
    print("\n" + "=" * 60)
    print("🎉 Cloud issues fixing complete!")
    print("=" * 60)
    
    print("\n🧪 Next steps:")
    print("1. Test minimal functionality: python3 test_minimal.py")
    print("2. Run fixed training: python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 3 --batch 1 --device cpu")
    print("3. If issues persist, check individual error messages above")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)