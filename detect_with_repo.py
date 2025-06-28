#!/usr/bin/env python3
"""
Detection script using the YOLOv13 repository directly
This script uses the ultralytics framework from the repository to detect objects.
"""

import sys
import os
from pathlib import Path

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

try:
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    import cv2
    import argparse
    
    def detect_objects(image_path, model_config="yolov13n", confidence=0.5, save_path=None):
        """
        Detect objects using YOLOv13 from the repository.
        
        Args:
            image_path (str): Path to input image
            model_config (str): Model configuration (yolov13n, yolov13s, yolov13l, yolov13x)
            confidence (float): Confidence threshold
            save_path (str, optional): Path to save result
        """
        # Use the YOLOv13 config file from the repository
        config_path = yolov13_path / "ultralytics" / "cfg" / "models" / "v13" / "yolov13.yaml"
        
        try:
            # Load YOLOv13 model with the config
            print(f"Loading YOLOv13 model from {config_path}...")
            model = YOLO(str(config_path))
            
            # Run inference
            print(f"Running detection on {image_path}...")
            results = model(image_path, conf=confidence)
            
            # Get the annotated image
            annotated_img = results[0].plot()
            
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
            print("This might be because pre-trained weights are not available.")
            print("The YOLOv13 repository contains the model architecture but may need trained weights.")
            return None, None
    
    def main():
        parser = argparse.ArgumentParser(description="YOLOv13 Detection using Repository")
        parser.add_argument("--image", type=str, required=True, help="Path to input image")
        parser.add_argument("--model", type=str, default="yolov13n", 
                          choices=["yolov13n", "yolov13s", "yolov13l", "yolov13x"],
                          help="Model size")
        parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
        parser.add_argument("--save", type=str, default=None, help="Path to save result")
        parser.add_argument("--show", action="store_true", help="Display result")
        
        args = parser.parse_args()
        
        # Check if image exists
        if not Path(args.image).exists():
            print(f"Error: Image {args.image} not found!")
            return
        
        # Run detection
        result_img, results = detect_objects(
            args.image, 
            args.model, 
            args.conf, 
            args.save
        )
        
        # Display if requested
        if args.show and result_img is not None:
            cv2.imshow("YOLOv13 Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

except ImportError as e:
    print(f"Import error: {e}")
    print("Could not import required modules from the YOLOv13 repository.")
    print("This repository might need additional setup or dependencies.")
    
    # Fallback: Try to use a simple detection approach
    print("\nTrying alternative approach...")
    
    def simple_detect_fallback(image_path):
        """Simple fallback detection using OpenCV if available."""
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return
            
            print(f"Loaded image: {image_path}")
            print(f"Image shape: {img.shape}")
            
            # For demonstration, just show the image with a message
            cv2.putText(img, "YOLOv13 detection requires trained weights", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Repository contains model architecture only", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save result
            output_path = "detection_demo.jpg"
            cv2.imwrite(output_path, img)
            print(f"Demo image saved to: {output_path}")
            
            # Show image
            cv2.imshow("YOLOv13 Repository Demo", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except ImportError:
            print("OpenCV not available. Cannot process image.")
    
    def main():
        parser = argparse.ArgumentParser(description="YOLOv13 Detection Fallback")
        parser.add_argument("--image", type=str, required=True, help="Path to input image")
        args = parser.parse_args()
        
        simple_detect_fallback(args.image)

if __name__ == "__main__":
    main()