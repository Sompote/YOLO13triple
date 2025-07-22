#!/usr/bin/env python3
"""
Hole Detection Inference Script
"""
import sys
from pathlib import Path
import argparse

# Setup local ultralytics
sys.path.insert(0, str(Path(__file__).parent / "yolov13"))

try:
    from ultralytics import YOLO
    print("✅ Successfully imported YOLO from local ultralytics")
except ImportError as e:
    print(f"❌ Failed to import YOLO: {e}")
    print("Make sure yolov13/ directory exists and contains ultralytics module")
    sys.exit(1)

def run_inference(model_path, source, save_dir="inference_results", conf_threshold=0.25):
    """Run inference with trained hole detection model"""
    
    print(f"🔮 Starting hole detection inference...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Source: {source}")
    print(f"📁 Save directory: {save_dir}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Verify source exists
    if not Path(source).exists():
        print(f"❌ Source not found: {source}")
        return False
    
    try:
        # Load model
        print("📥 Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Run inference
        print("🚀 Running inference...")
        results = model(
            source, 
            save=True, 
            project=save_dir,
            name="hole_detection",
            conf=conf_threshold,
            verbose=True
        )
        
        print(f"✅ Inference completed!")
        print(f"📊 Processed {len(results)} image(s)")
        print(f"💾 Results saved to: {save_dir}/hole_detection/")
        
        # Print detection summary
        for i, result in enumerate(results):
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"   Image {i+1}: {detections} hole(s) detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_model(model_path, data_config):
    """Validate model performance on dataset"""
    
    print(f"🧪 Validating model performance...")
    
    try:
        model = YOLO(model_path)
        metrics = model.val(data=data_config)
        
        print(f"📊 Model Performance Metrics:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall: {metrics.box.mr:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Hole Detection Inference')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True, 
                       help='Image file, folder, or video path')
    parser.add_argument('--save-dir', type=str, default='inference_results', 
                       help='Save directory for results')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation on dataset')
    parser.add_argument('--data', type=str, default='datatrain.yaml', 
                       help='Dataset config for validation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 YOLOv13 Hole Detection Inference")
    print("=" * 60)
    
    # Run validation if requested
    if args.validate:
        print("\n🧪 Running model validation...")
        validate_model(args.model, args.data)
        print()
    
    # Run inference
    success = run_inference(
        model_path=args.model,
        source=args.source,
        save_dir=args.save_dir,
        conf_threshold=args.conf
    )
    
    if success:
        print("\n🎉 Inference completed successfully!")
        print(f"📁 Check results in: {args.save_dir}/hole_detection/")
    else:
        print("\n❌ Inference failed")
        print("💡 Make sure:")
        print("   1. Model file exists and is valid")
        print("   2. Source image/folder exists")
        print("   3. yolov13/ directory contains ultralytics")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)