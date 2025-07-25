#!/usr/bin/env python3
"""
Triple YOLO Validation Script
Properly loads triple input models and runs validation
"""

import sys
import os
import yaml
from pathlib import Path

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

def run_validation(weights_path, data_config="datatrain.yaml"):
    """Run validation on triple input model"""
    
    print("🚀 YOLOv13 Triple Input Validation")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
        
    if not os.path.exists(data_config):
        print(f"❌ Error: Dataset config not found: {data_config}")
        return False
    
    # Setup triple environment
    if not setup_triple_environment():
        return False
    
    try:
        # Load dataset configuration
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        is_triple_input = config.get('triple_input', False)
        print(f"🔍 Dataset type: {'Triple Input' if is_triple_input else 'Single Input'}")
        print(f"📦 Model weights: {weights_path}")
        print(f"📊 Dataset config: {data_config}")
        
        # Import after environment setup
        from ultralytics import YOLO
        
        # Load model
        print("\n📦 Loading model...")
        model = YOLO(weights_path)
        
        # Create a temporary validation config that points to test split
        temp_config = config.copy()
        if 'test' in config:
            temp_config['val'] = config['test']  # Use test data for validation
            print("🎯 Using test split for validation")
        else:
            print("⚠️  No test split found, using val split")
        
        # Save temporary config
        temp_config_path = "temp_val_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        print("🎯 Running validation...")
        
        # Run validation
        results = model.val(
            data=temp_config_path,
            split='val',  # Use val split (which points to test data)
            save_json=True,
            plots=True,
            verbose=True,
            device='cpu'
        )
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        # Display results
        print("\n📊 Validation Results:")
        print("=" * 30)
        
        if hasattr(results, 'box') and results.box:
            box_results = results.box
            
            if hasattr(box_results, 'map50') and box_results.map50 is not None:
                print(f"mAP@0.5: {box_results.map50:.4f}")
            if hasattr(box_results, 'map') and box_results.map is not None:
                print(f"mAP@0.5:0.95: {box_results.map:.4f}")
            if hasattr(box_results, 'mp') and box_results.mp is not None:
                print(f"Precision: {box_results.mp:.4f}")
            if hasattr(box_results, 'mr') and box_results.mr is not None:
                print(f"Recall: {box_results.mr:.4f}")
                
            # Calculate F1 score
            if (hasattr(box_results, 'mp') and hasattr(box_results, 'mr') and 
                box_results.mp is not None and box_results.mr is not None):
                precision = float(box_results.mp)
                recall = float(box_results.mr)
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    print(f"F1-Score: {f1_score:.4f}")
        
        # Speed metrics
        if hasattr(results, 'speed') and results.speed:
            print(f"\n⚡ Speed Metrics:")
            speed = results.speed
            if isinstance(speed, dict):
                if 'preprocess' in speed:
                    print(f"Preprocessing: {speed['preprocess']:.2f} ms")
                if 'inference' in speed:
                    print(f"Inference: {speed['inference']:.2f} ms")
                if 'postprocess' in speed:
                    print(f"Postprocessing: {speed['postprocess']:.2f} ms")
                
                total_time = sum([v for v in speed.values() if v is not None])
                print(f"Total: {total_time:.2f} ms")
        
        print("\n✅ Validation completed successfully!")
        print("📁 Results saved to runs/val/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Clean up temp file if it exists
        temp_config_path = "temp_val_config.yaml"
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_validation.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python run_validation.py runs/unified_train_triple/yolo_s_triple49/weights/best.pt")
        print("  python run_validation.py best.pt datatrain.yaml")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    
    # Handle wildcard paths
    if "*" in weights_path:
        from glob import glob
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    success = run_validation(weights_path, data_config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()