#!/usr/bin/env python3
"""
Test script to verify local ultralytics import works correctly
"""

import sys
import os
from pathlib import Path

def test_local_ultralytics():
    """Test that local ultralytics can be imported successfully"""
    print("🧪 Testing local ultralytics import...")
    
    # Setup local ultralytics path
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        print(f"❌ YOLOv13 directory not found: {yolov13_path}")
        return False
    
    # Add yolov13 directory to Python path
    if str(yolov13_path) not in sys.path:
        sys.path.insert(0, str(yolov13_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", "")
    
    # Remove any existing ultralytics imports from cache
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"🗑️ Removed cached module: {module}")
    
    # Test import
    try:
        print("📥 Testing YOLO import...")
        from ultralytics import YOLO
        print("✅ Successfully imported YOLO")
        
        print("📥 Testing LOGGER import...")
        try:
            from ultralytics.utils import LOGGER
            print("✅ Successfully imported LOGGER")
        except ImportError as e:
            print(f"⚠️ LOGGER import failed: {e}")
        
        print("📥 Testing model creation...")
        # Test with existing config
        config_path = "yolov13s_standalone.yaml"
        if Path(config_path).exists():
            model = YOLO(config_path)
            print(f"✅ Successfully created model from: {config_path}")
        else:
            print(f"⚠️ Config file not found: {config_path}")
        
        # Test version
        try:
            import ultralytics
            print(f"✅ Ultralytics version: {ultralytics.__version__}")
        except AttributeError:
            print("⚠️ Could not get ultralytics version")
        
        # Test available models
        print("📋 Testing available model configs...")
        models_dir = yolov13_path / "ultralytics" / "cfg" / "models"
        if models_dir.exists():
            print(f"✅ Models directory found: {models_dir}")
            v13_dir = models_dir / "v13"
            if v13_dir.exists():
                configs = list(v13_dir.glob("*.yaml"))
                print(f"✅ Found {len(configs)} YOLOv13 configs:")
                for config in configs[:5]:  # Show first 5
                    print(f"   - {config.name}")
                if len(configs) > 5:
                    print(f"   ... and {len(configs) - 5} more")
            else:
                print("⚠️ YOLOv13 configs directory not found")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 YOLOv13 Local Import Test")
    print("=" * 60)
    
    success = test_local_ultralytics()
    
    print("=" * 60)
    if success:
        print("✅ All tests passed! Local ultralytics import is working correctly.")
        print("🚀 You can now run training scripts:")
        print("   python standalone_train_fixed.py --data your_dataset.yaml")
        print("   python train_triple_fixed.py --data your_dataset.yaml")
    else:
        print("❌ Tests failed. Please check the yolov13 directory and dependencies.")
        print("🔧 Make sure:")
        print("   - yolov13 directory exists in the current directory")
        print("   - yolov13/ultralytics directory contains the ultralytics module")
        print("   - All required dependencies are installed")
    print("=" * 60)

if __name__ == "__main__":
    main() 