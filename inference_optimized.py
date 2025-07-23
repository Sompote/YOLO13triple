#!/usr/bin/env python3
"""
Optimized YOLOv13 Inference Script
"""

import sys
import os
from pathlib import Path
import argparse

def setup_local_ultralytics():
    """Setup local ultralytics import"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        return None
    
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    os.environ.update({
        "PYTHONPATH": str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", ""),
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "ULTRALYTICS_OFFLINE": "1"
    })
    
    return yolov13_path

def run_inference(model_path, source, conf=0.01, iou=0.3, save=True, project="inference_results", name="detect"):
    """Run inference with optimized thresholds"""
    setup_local_ultralytics()
    
    try:
        from ultralytics import YOLO
        
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return False
        
        model = YOLO(model_path)
        
        results = model(source, 
                       conf=conf, 
                       iou=iou, 
                       save=save, 
                       project=project, 
                       name=name,
                       verbose=False)
        
        # Count detections
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
        
        print(f"Inference completed! Total detections: {total_detections}")
        if save:
            print(f"Results saved to: {project}/{name}/")
        
        return True
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Optimized YOLOv13 Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True, help='Source path (image/folder/video)')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold')
    parser.add_argument('--project', type=str, default='inference_results', help='Project folder')
    parser.add_argument('--name', type=str, default='detect', help='Run name')
    parser.add_argument('--nosave', action='store_true', help='Don\'t save results')
    
    args = parser.parse_args()
    
    success = run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=not args.nosave,
        project=args.project,
        name=args.name
    )
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)