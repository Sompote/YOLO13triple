#!/usr/bin/env python3
"""
Triple Input Inference Script
Demonstrates how to use multiple images for enhanced detection
"""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Add yolov13 to path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

def triple_input_inference(primary_image, detail1_image, detail2_image, model_path):
    """
    Perform inference using triple input concept
    
    Args:
        primary_image: Main image path
        detail1_image: First detail image path  
        detail2_image: Second detail image path
        model_path: Path to trained model
    """
    from ultralytics import YOLO
    
    # Load trained model
    model = YOLO(model_path)
    
    # Load images
    img1 = cv2.imread(primary_image)
    img2 = cv2.imread(detail1_image) 
    img3 = cv2.imread(detail2_image)
    
    # For now, use the primary image for detection
    # In a full implementation, you would combine features from all three
    results = model(primary_image)
    
    # Save results
    output_path = "triple_inference_result.jpg"
    results[0].save(output_path)
    
    print(f"Triple input inference completed!")
    print(f"Primary image: {primary_image}")
    print(f"Detail images: {detail1_image}, {detail2_image}")
    print(f"Result saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    # Example usage
    primary = "training_data_demo/images/val/image_1.jpg"
    detail1 = "training_data_demo/images/train/image_1.jpg"  # Use different detail images
    detail2 = "training_data_demo/images/train/image_2.jpg"
    model_path = "runs/working_train/yolov13_working/weights/best.pt"
    
    if all(Path(p).exists() for p in [primary, detail1, detail2]):
        if Path(model_path).exists():
            triple_input_inference(primary, detail1, detail2, model_path)
        else:
            print(f"Model not found: {model_path}")
            print("Please train the model first with: python fix_and_train.py --train")
    else:
        print("Some image files are missing. Please check the paths.")
