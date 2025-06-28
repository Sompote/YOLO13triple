#!/usr/bin/env python3
"""
Enhanced Triple Input YOLO Inference with Model Variants and Pretrained Weights
Supports YOLOv13n, YOLOv13s, YOLOv13m, YOLOv13l, YOLOv13x variants
"""

import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
import argparse
import time
from typing import List, Optional, Union

# Add project paths
project_root = Path(__file__).parent
yolov13_path = project_root / "yolov13"
sys.path.insert(0, str(yolov13_path))
sys.path.insert(0, str(project_root))

from models.triple_yolo_variants import (
    create_triple_yolo_model,
    triple_yolo13n, triple_yolo13s, triple_yolo13m, 
    triple_yolo13l, triple_yolo13x,
    MODEL_VARIANTS
)


class EnhancedTripleInference:
    """Enhanced inference class with model variant support."""
    
    def __init__(self, variant='n', weights=None, device='auto', nc=80, conf_thresh=0.25):
        """
        Initialize enhanced triple inference.
        
        Args:
            variant (str): Model variant ('n', 's', 'm', 'l', 'x')
            weights (str): Path to weights file or pretrained variant name
            device (str): Device to use ('auto', 'cpu', 'cuda', etc.)
            nc (int): Number of classes
            conf_thresh (float): Confidence threshold
        """
        self.variant = variant
        self.weights = weights
        self.nc = nc
        self.conf_thresh = conf_thresh
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing Enhanced Triple YOLO Inference")
        print(f"   Variant: YOLOv13{variant}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {nc}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Class names (COCO by default)
        self.class_names = self._get_class_names()
        
        print(f"✅ Model loaded successfully!")
    
    def _load_model(self):
        """Load model with specified variant and weights."""
        print(f"📦 Loading YOLOv13{self.variant} model...")
        
        # Create model based on variant
        if self.weights and ('yolov13' in str(self.weights) or Path(self.weights).exists()):
            # Load with pretrained weights
            model = create_triple_yolo_model(
                variant=self.variant,
                nc=self.nc,
                pretrained=self.weights
            )
        else:
            # Create without pretrained weights
            model = create_triple_yolo_model(
                variant=self.variant,
                nc=self.nc,
                pretrained=None
            )
            
            # Load custom weights if provided
            if self.weights and Path(self.weights).exists():
                print(f"Loading custom weights: {self.weights}")
                checkpoint = torch.load(self.weights, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.print_model_info()
        
        return model
    
    def _get_class_names(self):
        """Get class names (COCO dataset by default)."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ][:self.nc]
    
    def load_and_preprocess_images(self, primary_path, detail1_path, detail2_path, imgsz=640):
        """Load and preprocess triple images."""
        images = []
        paths = [primary_path, detail1_path, detail2_path]
        
        print(f"📸 Loading triple images...")
        for i, path in enumerate(paths):
            if not Path(path).exists():
                if i == 0:
                    raise FileNotFoundError(f"Primary image not found: {path}")
                else:
                    print(f"⚠️  Detail image {i} not found, using primary image")
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
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
            images.append(img_tensor)
        
        return images
    
    def run_inference(self, primary_path, detail1_path, detail2_path, imgsz=640, save_path=None):
        """
        Run inference on triple images.
        
        Args:
            primary_path (str): Path to primary image
            detail1_path (str): Path to detail image 1
            detail2_path (str): Path to detail image 2
            imgsz (int): Input image size
            save_path (str): Path to save result
        
        Returns:
            tuple: (detections, raw_predictions, inference_time)
        """
        # Load and preprocess images
        images = self.load_and_preprocess_images(primary_path, detail1_path, detail2_path, imgsz)
        
        print(f"🔍 Running inference...")
        start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            # Add batch dimension to each image
            batched_input = [img.unsqueeze(0) for img in images]
            predictions = self.model(batched_input)
        
        inference_time = time.time() - start_time
        
        # Post-process results
        detections = self._postprocess_predictions(predictions)
        
        print(f"⚡ Inference completed in {inference_time*1000:.1f}ms")
        print(f"🎯 Found {len(detections)} detections above confidence threshold {self.conf_thresh}")
        
        # Visualize and save results
        if save_path:
            result_img = self._visualize_results(primary_path, detections, save_path)
            return detections, predictions, inference_time, result_img
        
        return detections, predictions, inference_time
    
    def _postprocess_predictions(self, predictions):
        """Post-process model predictions."""
        if isinstance(predictions, tuple):
            pred = predictions[0]  # Inference output
            if isinstance(pred, torch.Tensor):
                pred = pred[0]  # First batch
            else:
                return []
        else:
            pred = predictions[0] if isinstance(predictions, list) else predictions
        
        # Apply confidence threshold
        if len(pred.shape) > 1 and pred.shape[-1] > 4:
            conf_mask = pred[..., 4] > self.conf_thresh
            filtered_pred = pred[conf_mask]
            
            # Convert to list of detections
            detections = []
            for detection in filtered_pred:
                x1, y1, x2, y2 = detection[:4].cpu().numpy()
                conf = detection[4].cpu().item()
                if len(detection) > 5:
                    cls_scores = detection[5:].cpu().numpy()
                    cls_id = np.argmax(cls_scores)
                else:
                    cls_id = 0
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                })
            
            return detections
        
        return []
    
    def _visualize_results(self, primary_path, detections, save_path):
        """Visualize detection results."""
        # Load primary image
        img = cv2.imread(primary_path)
        if img is None:
            print(f"Could not load image for visualization: {primary_path}")
            return None
        
        h, w = img.shape[:2]
        
        # Add model info
        cv2.putText(img, f"YOLOv13{self.variant} Triple Input Detection", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Device: {self.device} | Detections: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detections
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Scale coordinates to image size
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)
            
            # Choose color
            color = colors[det['class_id'] % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save result
        cv2.imwrite(save_path, img)
        print(f"💾 Result saved to: {save_path}")
        
        return img
    
    def benchmark_model(self, warmup_runs=5, benchmark_runs=20, imgsz=640):
        """Benchmark model performance."""
        print(f"🏃 Benchmarking YOLOv13{self.variant} performance...")
        
        # Create dummy inputs
        dummy_images = [torch.randn(1, 3, imgsz, imgsz).to(self.device) for _ in range(3)]
        
        # Warmup
        print(f"   Warming up ({warmup_runs} runs)...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_images)
        
        # Benchmark
        print(f"   Benchmarking ({benchmark_runs} runs)...")
        times = []
        with torch.no_grad():
            for _ in range(benchmark_runs):
                start_time = time.time()
                _ = self.model(dummy_images)
                times.append(time.time() - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time
        
        print(f"📊 Benchmark Results:")
        print(f"   Average inference time: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Min time: {min(times)*1000:.1f} ms")
        print(f"   Max time: {max(times)*1000:.1f} ms")
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'fps': fps,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced Triple Input YOLO Inference")
    
    # Input images
    parser.add_argument("--primary", type=str, help="Primary image path")
    parser.add_argument("--detail1", type=str, help="Detail image 1 path")
    parser.add_argument("--detail2", type=str, help="Detail image 2 path")
    
    # Model configuration
    parser.add_argument("--variant", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="Model variant (default: n)")
    parser.add_argument("--weights", type=str, help="Path to weights file or pretrained variant name")
    parser.add_argument("--nc", type=int, default=80, help="Number of classes (default: 80)")
    
    # Inference settings
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    
    # Output
    parser.add_argument("--save", type=str, help="Save path for result image")
    parser.add_argument("--show", action="store_true", help="Show result image")
    
    # Utilities
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--create-samples", action="store_true", help="Create sample images")
    parser.add_argument("--list-variants", action="store_true", help="List available model variants")
    
    args = parser.parse_args()
    
    # List variants
    if args.list_variants:
        print("📋 Available YOLOv13 Triple Input Variants:")
        for variant, config in MODEL_VARIANTS.items():
            print(f"   YOLOv13{variant}: depth={config['depth_multiple']}, "
                  f"width={config['width_multiple']}, max_channels={config['max_channels']}")
        return
    
    # Create sample images
    if args.create_samples:
        from detect_triple import create_sample_images
        sample_dir = create_sample_images()
        print(f"Sample images created in: {sample_dir}")
        print(f"Test with: python {__file__} --primary {sample_dir}/primary/image_1.jpg "
              f"--detail1 {sample_dir}/detail1/image_1.jpg --detail2 {sample_dir}/detail2/image_1.jpg")
        return
    
    # Initialize inference
    try:
        inference = EnhancedTripleInference(
            variant=args.variant,
            weights=args.weights,
            device=args.device,
            nc=args.nc,
            conf_thresh=args.conf
        )
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_results = inference.benchmark_model()
            return
        
        # Use sample data if no images specified
        if not args.primary:
            sample_dir = Path("sample_data")
            if sample_dir.exists():
                args.primary = str(sample_dir / "primary/image_1.jpg")
                args.detail1 = str(sample_dir / "detail1/image_1.jpg")
                args.detail2 = str(sample_dir / "detail2/image_1.jpg")
                print(f"Using sample images from {sample_dir}")
            else:
                print("❌ No images specified and no sample data found")
                print("Use --create-samples to generate test images")
                return
        
        # Validate image paths
        for path, name in [(args.primary, "primary"), (args.detail1, "detail1"), (args.detail2, "detail2")]:
            if not Path(path).exists():
                print(f"❌ {name} image not found: {path}")
                return
        
        # Set default save path
        if not args.save:
            args.save = f"result_yolov13{args.variant}_triple.jpg"
        
        # Run inference
        results = inference.run_inference(
            args.primary, args.detail1, args.detail2, 
            args.imgsz, args.save
        )
        
        if len(results) == 4:
            detections, predictions, inference_time, result_img = results
        else:
            detections, predictions, inference_time = results
            result_img = None
        
        # Print detection results
        if detections:
            print(f"\n🎯 Detection Results:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        # Show result if requested
        if args.show and result_img is not None:
            cv2.imshow(f"YOLOv13{args.variant} Triple Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"✅ Enhanced triple inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()