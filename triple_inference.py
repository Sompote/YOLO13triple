#!/usr/bin/env python3
"""
Triple Input Hole Detection Inference Script
Processes primary, detail1, and detail2 images for comprehensive hole detection
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
import cv2

# Setup local ultralytics
sys.path.insert(0, str(Path(__file__).parent / "yolov13"))

try:
    from ultralytics import YOLO
    print("✅ Successfully imported YOLO from local ultralytics")
except ImportError as e:
    print(f"❌ Failed to import YOLO: {e}")
    print("Make sure yolov13/ directory exists and contains ultralytics module")
    sys.exit(1)

def load_and_preprocess_image(image_path, target_size=640):
    """Load and preprocess image for inference"""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"✅ Loaded {image_path}: {img.shape}")
        return img
        
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None

def run_triple_inference(model_path, primary_img, detail1_img, detail2_img, 
                        save_dir="triple_inference_results", conf_threshold=0.25):
    """Run inference on triple input images"""
    
    print("🔮 Starting triple input hole detection inference...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Primary: {primary_img}")
    print(f"📁 Detail1: {detail1_img}")
    print(f"📁 Detail2: {detail2_img}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load model
        print("📥 Loading triple input model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Load and preprocess images
        print("\n📸 Loading triple input images...")
        primary = load_and_preprocess_image(primary_img)
        detail1 = load_and_preprocess_image(detail1_img)
        detail2 = load_and_preprocess_image(detail2_img)
        
        if primary is None or detail1 is None or detail2 is None:
            print("❌ Failed to load one or more images")
            return False
        
        # Create save directory
        save_path = Path(save_dir) / "triple_detection"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Run inference on each image type
        results = {}
        image_types = {
            'primary': primary_img,
            'detail1': detail1_img, 
            'detail2': detail2_img
        }
        
        print("\n🚀 Running inference on triple input set...")
        
        for img_type, img_path in image_types.items():
            print(f"\n🔍 Processing {img_type} image...")
            
            # Run inference
            result = model(
                img_path,
                save=True,
                project=str(save_path.parent),
                name=f"triple_detection/{img_type}",
                conf=conf_threshold,
                verbose=False
            )
            
            results[img_type] = result[0] if result else None
            
            # Print detection summary for this image
            if results[img_type] and results[img_type].boxes is not None:
                detections = len(results[img_type].boxes)
                print(f"   ✅ {img_type}: {detections} hole(s) detected")
                
                # Print confidence scores
                if detections > 0:
                    confidences = results[img_type].boxes.conf.cpu().numpy()
                    avg_conf = np.mean(confidences)
                    print(f"   📊 Average confidence: {avg_conf:.3f}")
            else:
                print(f"   📭 {img_type}: No holes detected")
        
        # Aggregate results across all three images
        print("\n📊 Triple Input Detection Summary:")
        print("=" * 50)
        
        total_detections = 0
        all_confidences = []
        
        for img_type, result in results.items():
            if result and result.boxes is not None:
                detections = len(result.boxes)
                total_detections += detections
                
                if detections > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    all_confidences.extend(confidences)
                    max_conf = np.max(confidences)
                    print(f"   {img_type:8}: {detections:2d} detections (max conf: {max_conf:.3f})")
                else:
                    print(f"   {img_type:8}: {detections:2d} detections")
            else:
                print(f"   {img_type:8}:  0 detections")
        
        print("-" * 50)
        print(f"   Total   : {total_detections:2d} detections across all images")
        
        if all_confidences:
            avg_conf = np.mean(all_confidences)
            max_conf = np.max(all_confidences)
            min_conf = np.min(all_confidences)
            print(f"   Confidence: avg={avg_conf:.3f}, max={max_conf:.3f}, min={min_conf:.3f}")
        
        # Create combined visualization if possible
        print(f"\n💾 Results saved to: {save_path}/")
        print(f"   - Primary results: {save_path}/primary/")
        print(f"   - Detail1 results: {save_path}/detail1/") 
        print(f"   - Detail2 results: {save_path}/detail2/")
        
        return True
        
    except Exception as e:
        print(f"❌ Triple inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_triple_inference(model_path, input_dir, save_dir="batch_triple_results", conf_threshold=0.25):
    """Run triple inference on a batch of image sets"""
    
    print("🔮 Starting batch triple input inference...")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Look for image sets (assuming naming convention: name_primary.jpg, name_detail1.jpg, name_detail2.jpg)
    primary_images = list(input_path.glob("*_primary.*"))
    
    if not primary_images:
        print(f"❌ No primary images found with pattern '*_primary.*' in {input_dir}")
        print("💡 Expected naming: basename_primary.jpg, basename_detail1.jpg, basename_detail2.jpg")
        return False
    
    print(f"📁 Found {len(primary_images)} image sets")
    
    success_count = 0
    
    for primary_img in primary_images:
        # Extract base name
        base_name = primary_img.stem.replace('_primary', '')
        
        # Find corresponding detail images
        detail1_img = primary_img.parent / f"{base_name}_detail1{primary_img.suffix}"
        detail2_img = primary_img.parent / f"{base_name}_detail2{primary_img.suffix}"
        
        if not detail1_img.exists() or not detail2_img.exists():
            print(f"⚠️ Missing detail images for {base_name}, skipping...")
            continue
        
        print(f"\n🔄 Processing image set: {base_name}")
        
        # Create individual save directory
        set_save_dir = Path(save_dir) / base_name
        
        # Run triple inference
        success = run_triple_inference(
            model_path=model_path,
            primary_img=str(primary_img),
            detail1_img=str(detail1_img),
            detail2_img=str(detail2_img),
            save_dir=str(set_save_dir),
            conf_threshold=conf_threshold
        )
        
        if success:
            success_count += 1
    
    print(f"\n✅ Batch processing completed: {success_count}/{len(primary_images)} sets processed successfully")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Triple Input Hole Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained triple input model (.pt file)')
    
    # Single set mode
    parser.add_argument('--primary', type=str,
                       help='Path to primary image')
    parser.add_argument('--detail1', type=str,
                       help='Path to detail1 image')
    parser.add_argument('--detail2', type=str,
                       help='Path to detail2 image')
    
    # Batch mode
    parser.add_argument('--batch-dir', type=str,
                       help='Directory containing image sets (batch processing)')
    
    parser.add_argument('--save-dir', type=str, default='triple_inference_results',
                       help='Save directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔮 YOLOv13 Triple Input Hole Detection Inference")
    print("=" * 70)
    
    # Validate arguments
    if args.batch_dir:
        # Batch mode
        success = batch_triple_inference(
            model_path=args.model,
            input_dir=args.batch_dir,
            save_dir=args.save_dir,
            conf_threshold=args.conf
        )
    elif args.primary and args.detail1 and args.detail2:
        # Single set mode
        success = run_triple_inference(
            model_path=args.model,
            primary_img=args.primary,
            detail1_img=args.detail1,
            detail2_img=args.detail2,
            save_dir=args.save_dir,
            conf_threshold=args.conf
        )
    else:
        print("❌ Please provide either:")
        print("   1. Single set: --primary, --detail1, --detail2")
        print("   2. Batch mode: --batch-dir")
        parser.print_help()
        return False
    
    if success:
        print("\n🎉 Triple input inference completed successfully!")
        print(f"📁 Check results in: {args.save_dir}/")
        print("\n💡 Triple input provides enhanced detection by analyzing:")
        print("   - Primary view: Overall context and main perspective")
        print("   - Detail1: Close-up details and fine features")
        print("   - Detail2: Additional angles and perspectives")
    else:
        print("\n❌ Triple inference failed")
        print("💡 Make sure:")
        print("   1. Model file exists and was trained with triple input")
        print("   2. All three image files exist and are valid")
        print("   3. Images correspond to the same object/scene")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)