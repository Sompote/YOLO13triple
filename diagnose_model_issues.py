#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify why the triple YOLO model
is not detecting objects and showing zero metrics
"""

import sys
import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import pandas as pd

def setup_triple_environment():
    """Setup environment for triple YOLO"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        print("❌ Error: yolov13 directory not found")
        return False
    
    # Clean imports and setup path
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    # Set environment
    os.environ.update({
        "PYTHONPATH": str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", ""),
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "ULTRALYTICS_OFFLINE": "1",
        "NUMPY_EXPERIMENTAL_DTYPE_API": "0"
    })
    
    return True

def diagnose_training_results(weights_path):
    """Analyze training results to identify issues"""
    print("🔍 Analyzing Training Results")
    print("=" * 40)
    
    # Find results.csv
    model_dir = Path(weights_path).parent.parent
    results_csv = model_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"❌ No results.csv found at: {results_csv}")
        return
    
    try:
        df = pd.read_csv(results_csv)
        print(f"📊 Training ran for {len(df)} epochs\n")
        
        # Check final metrics
        final_row = df.iloc[-1]
        print("📋 Final Training Metrics:")
        print(f"   Precision: {final_row.get('metrics/precision(B)', 0):.4f}")
        print(f"   Recall: {final_row.get('metrics/recall(B)', 0):.4f}")
        print(f"   mAP50: {final_row.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP75: {final_row.get('metrics/mAP75(B)', 0):.4f}")
        print(f"   mAP50-95: {final_row.get('metrics/mAP50-95(B)', 0):.4f}")
        
        # Check loss progression
        print(f"\n📉 Loss Analysis:")
        print(f"   Final Train Box Loss: {final_row.get('train/box_loss', 0):.4f}")
        print(f"   Final Train Cls Loss: {final_row.get('train/cls_loss', 0):.4f}")
        print(f"   Final Val Box Loss: {final_row.get('val/box_loss', 0):.4f}")
        print(f"   Final Val Cls Loss: {final_row.get('val/cls_loss', 0):.4f}")
        
        # Check if metrics ever improved
        max_map50 = df['metrics/mAP50(B)'].max()
        max_precision = df['metrics/precision(B)'].max()
        max_recall = df['metrics/recall(B)'].max()
        
        print(f"\n📈 Best Metrics During Training:")
        print(f"   Best mAP50: {max_map50:.4f}")
        print(f"   Best Precision: {max_precision:.4f}")
        print(f"   Best Recall: {max_recall:.4f}")
        
        if max_map50 == 0 and max_precision == 0 and max_recall == 0:
            print(f"\n❌ CRITICAL ISSUE: All metrics remained at zero throughout training!")
            print(f"   This indicates a fundamental problem with:")
            print(f"   1. Dataset format or labels")
            print(f"   2. Model architecture compatibility")
            print(f"   3. Triple input data loading")
            
    except Exception as e:
        print(f"❌ Error reading training results: {e}")

def diagnose_dataset_format(data_config):
    """Check dataset format and labels"""
    print(f"\n🔍 Analyzing Dataset Format")
    print("=" * 40)
    
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = Path(config['path'])
        print(f"📁 Dataset path: {base_path}")
        print(f"🏷️  Number of classes: {config.get('nc', 'Unknown')}")
        print(f"📝 Class names: {config.get('names', 'Unknown')}")
        print(f"🔀 Triple input: {config.get('triple_input', False)}")
        
        # Check train/val/test splits
        for split in ['train', 'val', 'test']:
            if split in config:
                split_path = base_path / config[split]
                images = glob(str(split_path / "*.jpg")) + glob(str(split_path / "*.png"))
                print(f"🖼️  {split.capitalize()} images: {len(images)} found")
                
                # Check corresponding labels
                labels_path = base_path / "labels" / "primary" / split
                labels = glob(str(labels_path / "*.txt"))
                print(f"🏷️  {split.capitalize()} labels: {len(labels)} found")
                
                # Sample a few labels to check format
                if labels:
                    sample_label = labels[0]
                    with open(sample_label, 'r') as f:
                        lines = f.readlines()
                    print(f"📋 Sample label ({Path(sample_label).name}): {len(lines)} objects")
                    if lines:
                        print(f"    First object: {lines[0].strip()}")
        
        # Check triple input directories
        if config.get('triple_input'):
            detail1_path = base_path / config.get('detail1_path', 'images/detail1')
            detail2_path = base_path / config.get('detail2_path', 'images/detail2')
            
            print(f"\n🔍 Triple Input Directories:")
            print(f"   Detail1 exists: {detail1_path.exists()}")
            print(f"   Detail2 exists: {detail2_path.exists()}")
            
            if detail1_path.exists():
                detail1_images = glob(str(detail1_path / "*" / "*.jpg")) + glob(str(detail1_path / "*" / "*.png"))
                print(f"   Detail1 images: {len(detail1_images)}")
            
            if detail2_path.exists():
                detail2_images = glob(str(detail2_path / "*" / "*.jpg")) + glob(str(detail2_path / "*" / "*.png"))
                print(f"   Detail2 images: {len(detail2_images)}")
                
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")

def diagnose_model_architecture(weights_path):
    """Check model architecture and compatibility"""
    print(f"\n🔍 Analyzing Model Architecture")
    print("=" * 40)
    
    if not setup_triple_environment():
        return
    
    try:
        from ultralytics import YOLO
        
        print(f"📦 Loading model: {Path(weights_path).name}")
        model = YOLO(weights_path)
        
        # Get model info
        print(f"🏗️  Model architecture: {model.model.__class__.__name__}")
        
        # Check input channels
        if hasattr(model.model, 'model') and len(model.model.model) > 0:
            first_layer = model.model.model[0]
            if hasattr(first_layer, 'conv'):
                conv_layer = first_layer.conv
                if hasattr(conv_layer, 'in_channels'):
                    input_channels = conv_layer.in_channels
                    print(f"🔢 Input channels: {input_channels}")
                    
                    if input_channels == 9:
                        print(f"✅ Model expects 9 channels (triple input)")
                    elif input_channels == 3:
                        print(f"⚠️  Model expects 3 channels (single input)")
                    else:
                        print(f"❓ Unexpected input channels: {input_channels}")
        
        # Try to get model summary
        try:
            print(f"\n📊 Model Summary:")
            model.info(verbose=False)
        except:
            print(f"⚠️  Could not generate model summary")
            
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")

def diagnose_image_compatibility(data_config):
    """Test if images can be loaded properly for triple input"""
    print(f"\n🔍 Testing Image Loading Compatibility")
    print("=" * 40)
    
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = Path(config['path'])
        test_images_dir = base_path / config.get('test', config.get('val', 'images/primary/val'))
        detail1_dir = base_path / config.get('detail1_path', 'images/detail1')
        detail2_dir = base_path / config.get('detail2_path', 'images/detail2')
        
        # Get a test image
        test_images = glob(str(test_images_dir / "*.jpg")) + glob(str(test_images_dir / "*.png"))
        if not test_images:
            print(f"❌ No test images found in {test_images_dir}")
            return
        
        test_image = test_images[0]
        image_name = Path(test_image).name
        
        print(f"🖼️  Testing with: {image_name}")
        
        # Try to load primary image
        primary_img = cv2.imread(test_image)
        if primary_img is None:
            print(f"❌ Could not load primary image")
            return
        
        print(f"✅ Primary image loaded: {primary_img.shape}")
        
        # Try to load detail images
        detail1_path = detail1_dir / "test" / image_name
        detail2_path = detail2_dir / "test" / image_name
        
        detail1_img = cv2.imread(str(detail1_path)) if detail1_path.exists() else None
        detail2_img = cv2.imread(str(detail2_path)) if detail2_path.exists() else None
        
        print(f"📸 Detail1 image: {'✅ Loaded' if detail1_img is not None else '⚠️  Using primary as fallback'}")
        print(f"📸 Detail2 image: {'✅ Loaded' if detail2_img is not None else '⚠️  Using primary as fallback'}")
        
        # Test triple image creation
        if detail1_img is None:
            detail1_img = primary_img.copy()
        if detail2_img is None:
            detail2_img = primary_img.copy()
        
        # Ensure same size
        h, w = primary_img.shape[:2]
        detail1_img = cv2.resize(detail1_img, (w, h))
        detail2_img = cv2.resize(detail2_img, (w, h))
        
        print(f"🔧 Image sizes standardized to: {w}x{h}")
        
        # Test concatenation
        try:
            triple_img = np.concatenate([primary_img, detail1_img, detail2_img], axis=2)
            print(f"✅ Triple image created: {triple_img.shape} (should be HxWx9)")
            
            if triple_img.shape[2] == 9:
                print(f"✅ Triple input format is correct")
            else:
                print(f"❌ Triple input format is incorrect")
                
        except Exception as e:
            print(f"❌ Error creating triple image: {e}")
            
    except Exception as e:
        print(f"❌ Error testing image compatibility: {e}")

def generate_diagnostic_summary():
    """Generate comprehensive diagnostic summary"""
    print(f"\n📋 Diagnostic Summary & Recommendations")
    print("=" * 50)
    
    print(f"🔍 Issues Identified:")
    print(f"   1. Training metrics remained at zero throughout training")
    print(f"   2. Model is not detecting any objects during inference")
    print(f"   3. Validation pipeline shows IndexError")
    
    print(f"\n💡 Potential Causes:")
    print(f"   1. Label format issues (incorrect YOLO format)")
    print(f"   2. Image-label mismatch (missing labels for images)")
    print(f"   3. Triple input data loading problems")
    print(f"   4. Model architecture issues")
    print(f"   5. Training configuration problems")
    
    print(f"\n🔧 Recommended Actions:")
    print(f"   1. Verify label files are in correct YOLO format")
    print(f"   2. Check that all training images have corresponding labels")
    print(f"   3. Test with single-input mode first")
    print(f"   4. Reduce training complexity (smaller batch, lower learning rate)")
    print(f"   5. Try training with augmentations enabled")
    print(f"   6. Verify the triple dataset implementation")

def main():
    """Main diagnostic function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python diagnose_model_issues.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python diagnose_model_issues.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python diagnose_model_issues.py best.pt datatrain.yaml")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    
    # Handle wildcard paths
    if "*" in weights_path:
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    print("🚀 YOLOv13 Triple Input Model Diagnostics")
    print("=" * 60)
    print(f"📦 Model: {weights_path}")
    print(f"📊 Dataset: {data_config}")
    
    # Run all diagnostic checks
    diagnose_training_results(weights_path)
    diagnose_dataset_format(data_config)
    diagnose_model_architecture(weights_path)
    diagnose_image_compatibility(data_config)
    generate_diagnostic_summary()
    
    print(f"\n✅ Diagnostic analysis completed!")

if __name__ == "__main__":
    main()