#!/usr/bin/env python3
"""
Triple YOLO Test Script - Compatible with Triple Input Models
Uses the same training pipeline to ensure proper dataset handling
"""

import sys
import os
import yaml
from pathlib import Path

# Add yolov13 path to system path
current_dir = Path(__file__).parent
yolov13_path = current_dir / "yolov13"
if yolov13_path.exists():
    sys.path.insert(0, str(yolov13_path))

def test_triple_model(weights_path, data_config="datatrain.yaml"):
    """
    Test triple YOLO model using the training pipeline for compatibility
    """
    
    print("🚀 Testing YOLOv13 Triple Input Model")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
        
    if not os.path.exists(data_config):
        print(f"❌ Error: Dataset config not found: {data_config}")
        return False
    
    try:
        # Load dataset configuration
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        is_triple_input = config.get('triple_input', False)
        print(f"🔍 Dataset type: {'Triple Input' if is_triple_input else 'Single Input'}")
        print(f"📦 Model weights: {weights_path}")
        print(f"📊 Dataset config: {data_config}")
        
        if not is_triple_input:
            print("⚠️  Warning: Dataset config doesn't specify triple_input=true")
            print("💡 For standard YOLO models, use: python test_model.py")
        
        # Import the training modules that handle triple datasets
        try:
            from ultralytics import YOLO
            from ultralytics.data.build import build_dataloader
            from ultralytics.data.triple_dataset import TripleYOLODataset
            print("✅ Triple dataset modules imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
            print("💡 Make sure yolov13/ultralytics contains the triple_dataset.py module")
            return False
        
        # Load model
        print("\n📦 Loading trained model...")
        model = YOLO(weights_path)
        
        # Run validation using the model's built-in validation
        # This should work because during training, the model was configured for triple input
        print("🎯 Running validation on test dataset...")
        print("⚡ Using triple input validation pipeline...")
        
        results = model.val(
            data=data_config,
            split='test',
            save_json=True,
            plots=True,
            verbose=True,
            device='cpu'  # Force CPU to avoid CUDA issues
        )
        
        # Display results
        print("\n📊 Test Results Summary:")
        print("=" * 40)
        
        if hasattr(results.box, 'map50') and results.box.map50 is not None:
            print(f"mAP@0.5: {results.box.map50:.4f}")
        if hasattr(results.box, 'map') and results.box.map is not None:
            print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        if hasattr(results.box, 'mp') and results.box.mp is not None:
            print(f"Precision: {results.box.mp:.4f}")
        if hasattr(results.box, 'mr') and results.box.mr is not None:
            print(f"Recall: {results.box.mr:.4f}")
            
        # Calculate F1 score
        if hasattr(results.box, 'mp') and hasattr(results.box, 'mr'):
            precision = float(results.box.mp) if results.box.mp is not None else 0
            recall = float(results.box.mr) if results.box.mr is not None else 0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                print(f"F1-Score: {f1_score:.4f}")
        
        # Speed metrics
        if hasattr(results, 'speed') and results.speed:
            print(f"\n⚡ Speed Metrics:")
            if 'preprocess' in results.speed:
                print(f"Preprocessing: {results.speed['preprocess']:.2f} ms")
            if 'inference' in results.speed:
                print(f"Inference: {results.speed['inference']:.2f} ms") 
            if 'postprocess' in results.speed:
                print(f"Postprocessing: {results.speed['postprocess']:.2f} ms")
            
            total_time = sum([v for v in results.speed.values() if v is not None])
            print(f"Total: {total_time:.2f} ms")
        
        print("\n✅ Testing completed successfully!")
        print(f"📁 Detailed results saved to: runs/val/")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Provide specific guidance for common errors
        if "expected input" in str(e) and "channels" in str(e):
            print("\n💡 Channel mismatch detected!")
            print("   This usually means:")
            print("   1. The model was trained with triple input (9 channels)")
            print("   2. But validation is using standard dataset (3 channels)")
            print("   3. Try: python unified_train_optimized.py --data datatrain.yaml --task val --model weights.pt")
        
        return False

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_triple_model.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python test_triple_model.py runs/unified_train_triple/yolo_s_triple49/weights/best.pt")
        print("  python test_triple_model.py best.pt datatrain.yaml")
        print("  python test_triple_model.py \"runs/*/weights/best.pt\"")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    
    # Handle wildcard paths
    if "*" in weights_path:
        from glob import glob
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]  # Use first match
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    success = test_triple_model(weights_path, data_config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()