#!/usr/bin/env python3
"""
Direct inference script for triple input YOLO model
Bypasses the standard YOLO predictor for better control over triple inputs
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import cv2
import numpy as np

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

from ultralytics.nn.modules.conv import TripleInputConv, Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules import Concat
from ultralytics.utils.torch_utils import initialize_weights


class TripleYOLOModel(nn.Module):
    """Simple triple input YOLO model for testing."""
    
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        
        # Backbone with triple input
        self.conv0 = TripleInputConv(3, 64, 3, 2)  # Triple input layer
        self.conv1 = Conv(64, 128, 3, 2)
        self.c2f1 = C2f(128, 128, 3)
        self.conv2 = Conv(128, 256, 3, 2)
        self.c2f2 = C2f(256, 256, 6)
        self.conv3 = Conv(256, 512, 3, 2)
        self.c2f3 = C2f(512, 512, 6)
        self.conv4 = Conv(512, 1024, 3, 2)
        self.c2f4 = C2f(1024, 1024, 3)
        self.sppf = SPPF(1024, 1024, 5)
        
        # Head
        self.upsample1 = nn.Upsample(None, 2, "nearest")
        self.concat1 = Concat(1)
        self.c2f_head1 = C2f(1024 + 512, 512, 3)
        
        self.upsample2 = nn.Upsample(None, 2, "nearest")
        self.concat2 = Concat(1)
        self.c2f_head2 = C2f(512 + 256, 256, 3)
        
        self.conv_head1 = Conv(256, 256, 3, 2)
        self.concat3 = Concat(1)
        self.c2f_head3 = C2f(256 + 512, 512, 3)
        
        self.conv_head2 = Conv(512, 512, 3, 2)
        self.concat4 = Concat(1)
        self.c2f_head4 = C2f(512 + 1024, 1024, 3)
        
        # Detection heads
        self.detect = Detect(nc, [256, 512, 1024])
        
        # Initialize weights
        initialize_weights(self)
    
    def forward(self, x):
        """Forward pass with triple input support."""
        # Backbone
        x0 = self.conv0(x)  # Triple input processing
        x1 = self.conv1(x0)
        x1 = self.c2f1(x1)
        x2 = self.conv2(x1)
        x2 = self.c2f2(x2)  # P3
        x3 = self.conv3(x2)
        x3 = self.c2f3(x3)  # P4
        x4 = self.conv4(x3)
        x4 = self.c2f4(x4)
        x4 = self.sppf(x4)  # P5
        
        # Head
        p5 = x4
        p4_up = self.upsample1(p5)
        p4 = self.concat1([p4_up, x3])
        p4 = self.c2f_head1(p4)
        
        p3_up = self.upsample2(p4)
        p3 = self.concat2([p3_up, x2])
        p3 = self.c2f_head2(p3)  # Small objects
        
        # Downsample path
        p4_down = self.conv_head1(p3)
        p4 = self.concat3([p4_down, p4])
        p4 = self.c2f_head3(p4)  # Medium objects
        
        p5_down = self.conv_head2(p4)
        p5 = self.concat4([p5_down, p5])
        p5 = self.c2f_head4(p5)  # Large objects
        
        # Detection
        return self.detect([p3, p4, p5])


def load_and_preprocess_images(primary_path, detail1_path, detail2_path, imgsz=640):
    """Load and preprocess triple images."""
    images = []
    paths = [primary_path, detail1_path, detail2_path]
    
    for i, path in enumerate(paths):
        if not Path(path).exists():
            if i == 0:
                raise FileNotFoundError(f"Primary image not found: {path}")
            else:
                print(f"Warning: Detail image {i} not found, using primary image")
                path = primary_path
        
        # Load image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        
        # Resize
        img = cv2.resize(img, (imgsz, imgsz))
        
        # Convert BGR to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        images.append(img_tensor)
    
    return images


def triple_inference(primary_path, detail1_path, detail2_path, 
                    imgsz=640, conf_threshold=0.25):
    """Run inference with triple input model."""
    
    # Load model
    print("Creating triple input YOLO model...")
    model = TripleYOLOModel(nc=80)
    model.eval()
    
    # Load and preprocess images
    print("Loading triple images...")
    images = load_and_preprocess_images(primary_path, detail1_path, detail2_path, imgsz)
    
    # Prepare input (batch size 1)
    input_tensor = images  # Pass as list for triple input
    
    print(f"Input shapes: {[img.shape for img in input_tensor]}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Add batch dimension to each image
        batched_input = [img.unsqueeze(0) for img in input_tensor]
        predictions = model(batched_input)
    
    print(f"Predictions type: {type(predictions)}")
    if isinstance(predictions, tuple):
        print(f"Predictions length: {len(predictions)}")
        for i, p in enumerate(predictions):
            if isinstance(p, torch.Tensor):
                print(f"  Prediction {i} shape: {p.shape}")
            elif isinstance(p, list):
                print(f"  Prediction {i} is list with {len(p)} elements")
                for j, item in enumerate(p):
                    if isinstance(item, torch.Tensor):
                        print(f"    Item {j} shape: {item.shape}")
            else:
                print(f"  Prediction {i} type: {type(p)}")
    else:
        print(f"Predictions shape: {predictions.shape}")
    
    print(f"Model completed inference successfully!")
    
    # Post-process results (basic)
    if isinstance(predictions, tuple):
        # YOLO detection head returns (inference_out, training_out)
        # Use inference output (first element)
        pred = predictions[0]  # Inference output
        if isinstance(pred, torch.Tensor):
            pred = pred[0]  # First batch
        else:
            print("Unexpected prediction format")
            return None, predictions
    else:
        pred = predictions[0]  # First batch
    
    # Apply confidence threshold (assuming format [x, y, w, h, conf, ...])
    if len(pred.shape) > 1 and pred.shape[-1] > 4:
        conf_mask = pred[..., 4] > conf_threshold
        filtered_pred = pred[conf_mask]
    else:
        print("Predictions format not as expected, returning raw predictions")
        filtered_pred = pred
    
    print(f"Detections after confidence filtering: {len(filtered_pred)}")
    
    return filtered_pred, predictions


def visualize_results(primary_path, predictions, save_path=None):
    """Visualize detection results on primary image."""
    
    # Load primary image
    img = cv2.imread(primary_path)
    if img is None:
        print(f"Could not load image for visualization: {primary_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Add text overlay
    cv2.putText(img, "Triple Input YOLO Detection", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Model: Custom Triple YOLO", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Detections: {len(predictions) if predictions is not None else 0}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw bounding boxes (if any predictions)
    if predictions is not None and len(predictions) > 0:
        for det in predictions:
            x1, y1, x2, y2 = det[:4].int().tolist()
            conf = det[4].item()
            # Scale coordinates to image size
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save result
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Result saved to: {save_path}")
    
    return img


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Triple Input YOLO Inference")
    parser.add_argument("--primary", type=str, required=True, help="Primary image path")
    parser.add_argument("--detail1", type=str, required=True, help="Detail image 1 path")
    parser.add_argument("--detail2", type=str, required=True, help="Detail image 2 path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="triple_inference_result.jpg", help="Save path")
    parser.add_argument("--show", action="store_true", help="Show result")
    
    args = parser.parse_args()
    
    try:
        # Run inference
        detections, raw_pred = triple_inference(
            args.primary, args.detail1, args.detail2,
            args.imgsz, args.conf
        )
        
        # Visualize results
        result_img = visualize_results(args.primary, detections, args.save)
        
        # Show if requested
        if args.show and result_img is not None:
            cv2.imshow("Triple Input YOLO Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("✅ Triple input inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()