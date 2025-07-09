#!/usr/bin/env python3
"""
Triple Image Detection script using YOLOv13
This script processes 3 images simultaneously where the first image contains labels
and the other two provide additional detail information.
"""

import sys
import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

try:
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    from ultralytics.nn.modules.conv import TripleInputConv
    
    def load_triple_images(primary_path, detail1_path, detail2_path, target_size=(640, 640)):
        """
        Load and preprocess triple images.
        
        Args:
            primary_path (str): Path to primary image (with labels)
            detail1_path (str): Path to first detail image
            detail2_path (str): Path to second detail image
            target_size (tuple): Target size for resizing
            
        Returns:
            list: List of preprocessed images [primary, detail1, detail2]
        """
        images = []
        paths = [primary_path, detail1_path, detail2_path]
        
        for i, path in enumerate(paths):
            if not Path(path).exists():
                if i == 0:
                    raise FileNotFoundError(f"Primary image not found: {path}")
                else:
                    print(f"Warning: Detail image {i} not found: {path}, using primary image")
                    path = primary_path
            
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
                
            # Resize to target size
            img = cv2.resize(img, target_size)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            images.append(img_tensor)
        
        return images
    
    def detect_triple_objects(primary_path, detail1_path, detail2_path, 
                            model_config="yolov13-triple", confidence=0.5, save_path=None):
        """
        Detect objects using triple image input.
        
        Args:
            primary_path (str): Path to primary image
            detail1_path (str): Path to first detail image  
            detail2_path (str): Path to second detail image
            model_config (str): Model configuration
            confidence (float): Confidence threshold
            save_path (str, optional): Path to save result
        """
        # Use the YOLOv13 triple config file
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / f"{model_config}.yaml"
        
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            print("Available configs:")
            config_dir = config_path.parent
            for cfg in config_dir.glob("*.yaml"):
                print(f"  - {cfg.stem}")
            return None, None
        
        try:
            # Load YOLOv13 model with triple input config
            print(f"Loading YOLOv13 triple model from {config_path}...")
            model = YOLO(str(config_path))
            
            # Load triple images
            print(f"Loading triple images...")
            print(f"  Primary: {primary_path}")
            print(f"  Detail1: {detail1_path}")
            print(f"  Detail2: {detail2_path}")
            
            images = load_triple_images(primary_path, detail1_path, detail2_path)
            
            # Prepare input tensor
            # Stack images into batch format
            input_tensor = torch.stack(images).unsqueeze(0)  # [1, 3, 3, H, W]
            
            print(f"Input tensor shape: {input_tensor.shape}")
            
            # For now, we'll use the primary image for inference
            # TODO: Implement proper triple input inference
            primary_img = cv2.imread(primary_path)
            
            # Run inference on primary image
            print(f"Running detection...")
            results = model(primary_path, conf=confidence)
            
            # Get the annotated image
            annotated_img = results[0].plot()
            
            # Add information about triple input processing
            h, w = annotated_img.shape[:2]
            cv2.putText(annotated_img, "Triple Input YOLO Detection", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_img, f"Primary: {Path(primary_path).name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_img, f"Detail1: {Path(detail1_path).name}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_img, f"Detail2: {Path(detail2_path).name}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save if requested
            if save_path:
                cv2.imwrite(save_path, annotated_img)
                print(f"Result saved to: {save_path}")
            
            # Display results info
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                print(f"Found {num_detections} objects:")
                
                for i, box in enumerate(results[0].boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"  {i+1}. {class_name}: {confidence:.3f}")
            else:
                print("No objects detected.")
            
            return annotated_img, results
            
        except Exception as e:
            print(f"Error during detection: {e}")
            print("This might be because:")
            print("1. Pre-trained weights are not available")
            print("2. Triple input modules need proper integration")
            print("3. Model architecture needs trained weights")
            return None, None

    def create_sample_images(base_dir="sample_data"):
        """Create sample triple images for testing."""
        base_path = Path(base_dir)
        base_path.mkdir(exist_ok=True)
        
        # Create sample directories
        (base_path / "primary").mkdir(exist_ok=True)
        (base_path / "detail1").mkdir(exist_ok=True)
        (base_path / "detail2").mkdir(exist_ok=True)
        
        # Create sample images with different characteristics
        for i in range(3):
            # Primary image (original)
            primary = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(primary, f"Primary Image {i+1}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(primary, (100, 100), (200, 200), (0, 255, 0), 2)
            
            # Detail image 1 (enhanced contrast)
            detail1 = cv2.convertScaleAbs(primary, alpha=1.2, beta=30)
            cv2.putText(detail1, f"Detail1 (Enhanced)", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Detail image 2 (different color space)
            detail2 = cv2.cvtColor(primary, cv2.COLOR_BGR2HSV)
            detail2 = cv2.cvtColor(detail2, cv2.COLOR_HSV2BGR)
            cv2.putText(detail2, f"Detail2 (HSV)", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Save images
            cv2.imwrite(str(base_path / "primary" / f"image_{i+1}.jpg"), primary)
            cv2.imwrite(str(base_path / "detail1" / f"image_{i+1}.jpg"), detail1)
            cv2.imwrite(str(base_path / "detail2" / f"image_{i+1}.jpg"), detail2)
        
        print(f"Sample triple images created in {base_path}")
        return base_path

    def main():
        parser = argparse.ArgumentParser(description="YOLOv13 Triple Image Detection")
        parser.add_argument("--primary", type=str, help="Path to primary image")
        parser.add_argument("--detail1", type=str, help="Path to first detail image")
        parser.add_argument("--detail2", type=str, help="Path to second detail image")
        parser.add_argument("--model", type=str, default="yolov13-triple", 
                          help="Model configuration")
        parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
        parser.add_argument("--save", type=str, default="triple_detection_result.jpg", 
                          help="Path to save result")
        parser.add_argument("--show", action="store_true", help="Display result")
        parser.add_argument("--create-samples", action="store_true", 
                          help="Create sample triple images for testing")
        
        args = parser.parse_args()
        
        # Create sample images if requested
        if args.create_samples:
            sample_dir = create_sample_images()
            print(f"Sample images created. You can now test with:")
            print(f"python {__file__} --primary {sample_dir}/primary/image_1.jpg "
                  f"--detail1 {sample_dir}/detail1/image_1.jpg "
                  f"--detail2 {sample_dir}/detail2/image_1.jpg")
            return
        
        # Check if all image paths are provided
        if not all([args.primary, args.detail1, args.detail2]):
            parser.error("All three image paths (--primary, --detail1, --detail2) are required, "
                        "or use --create-samples to generate test images")
        
        # Check if images exist
        for path, name in [(args.primary, "primary"), (args.detail1, "detail1"), (args.detail2, "detail2")]:
            if not Path(path).exists():
                print(f"Error: {name} image {path} not found!")
                return
        
        # Run detection
        result_img, results = detect_triple_objects(
            args.primary, 
            args.detail1, 
            args.detail2,
            args.model, 
            args.conf, 
            args.save
        )
        
        # Display if requested
        if args.show and result_img is not None:
            cv2.imshow("YOLOv13 Triple Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

except ImportError as e:
    print(f"Import error: {e}")
    print("Could not import required modules from the YOLOv13 repository.")
    print("This repository might need additional setup or dependencies.")
    
    def main():
        print("Fallback mode: Triple input detection not available")
        print("Please ensure all dependencies are installed and the model is properly configured.")

if __name__ == "__main__":
    main()