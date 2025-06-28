#!/usr/bin/env python3
"""
Test the trained triple input model
"""

import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

from triple_inference import TripleYOLOModel, load_and_preprocess_images, visualize_results


def load_trained_model(checkpoint_path, device='cpu'):
    """Load the trained model from checkpoint."""
    print(f"Loading trained model from: {checkpoint_path}")
    
    # Create model
    model = TripleYOLOModel(nc=80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model


def test_trained_inference(model, primary_path, detail1_path, detail2_path, 
                          imgsz=640, conf_threshold=0.25, device='cpu'):
    """Test inference with trained model."""
    
    # Load and preprocess images
    print("Loading triple images...")
    images = load_and_preprocess_images(primary_path, detail1_path, detail2_path, imgsz)
    
    print(f"Input shapes: {[img.shape for img in images]}")
    
    # Run inference
    print("Running inference with trained model...")
    with torch.no_grad():
        # Add batch dimension to each image
        batched_input = [img.unsqueeze(0).to(device) for img in images]
        predictions = model(batched_input)
    
    print(f"Predictions type: {type(predictions)}")
    if isinstance(predictions, tuple):
        print(f"Predictions length: {len(predictions)}")
        for i, p in enumerate(predictions):
            if isinstance(p, torch.Tensor):
                print(f"  Prediction {i} shape: {p.shape}")
            elif isinstance(p, list):
                print(f"  Prediction {i} is list with {len(p)} elements")
    
    # Post-process results
    if isinstance(predictions, tuple):
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


def compare_models(primary_path, detail1_path, detail2_path):
    """Compare untrained vs trained model performance."""
    device = torch.device("cpu")
    
    print("=" * 60)
    print("COMPARING UNTRAINED VS TRAINED MODEL")
    print("=" * 60)
    
    # Test untrained model
    print("\n1. Testing UNTRAINED model:")
    print("-" * 40)
    untrained_model = TripleYOLOModel(nc=80)
    untrained_model.to(device)
    untrained_model.eval()
    
    untrained_detections, _ = test_trained_inference(
        untrained_model, primary_path, detail1_path, detail2_path, device=device
    )
    
    # Test trained model
    print("\n2. Testing TRAINED model:")
    print("-" * 40)
    checkpoint_path = "runs/train_direct/best.pt"
    if Path(checkpoint_path).exists():
        trained_model = load_trained_model(checkpoint_path, device)
        trained_detections, _ = test_trained_inference(
            trained_model, primary_path, detail1_path, detail2_path, device=device
        )
        
        # Visualize results
        print("\n3. Generating comparison visualizations:")
        print("-" * 40)
        
        # Untrained result
        untrained_img = visualize_results(
            primary_path, untrained_detections, "untrained_result.jpg"
        )
        
        # Trained result  
        trained_img = visualize_results(
            primary_path, trained_detections, "trained_result.jpg"
        )
        
        print("Results saved:")
        print("  - untrained_result.jpg (untrained model)")
        print("  - trained_result.jpg (trained model)")
        
        # Summary
        print("\n4. SUMMARY:")
        print("-" * 40)
        print(f"Untrained model detections: {len(untrained_detections) if untrained_detections is not None else 0}")
        print(f"Trained model detections: {len(trained_detections) if trained_detections is not None else 0}")
        
        if trained_detections is not None and untrained_detections is not None:
            if len(trained_detections) != len(untrained_detections):
                print("✅ Training has changed the model's detection behavior!")
            else:
                print("ℹ️  Same number of detections, but confidence scores may have changed")
        
    else:
        print(f"❌ Trained model not found at {checkpoint_path}")
        print("Please run training first: python train_direct_triple.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Trained Triple Input Model")
    parser.add_argument("--primary", type=str, help="Primary image path")
    parser.add_argument("--detail1", type=str, help="Detail image 1 path")
    parser.add_argument("--detail2", type=str, help="Detail image 2 path")
    parser.add_argument("--checkpoint", type=str, default="runs/train_direct/best.pt", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="trained_model_result.jpg", help="Save path")
    parser.add_argument("--compare", action="store_true", help="Compare untrained vs trained")
    
    args = parser.parse_args()
    
    # Use sample data if no images specified
    if not args.primary:
        args.primary = "sample_data/primary/image_1.jpg"
        args.detail1 = "sample_data/detail1/image_1.jpg"
        args.detail2 = "sample_data/detail2/image_1.jpg"
        print(f"Using sample images:")
        print(f"  Primary: {args.primary}")
        print(f"  Detail1: {args.detail1}")
        print(f"  Detail2: {args.detail2}")
    
    # Check if images exist
    for path, name in [(args.primary, "primary"), (args.detail1, "detail1"), (args.detail2, "detail2")]:
        if not Path(path).exists():
            print(f"❌ {name} image not found: {path}")
            return
    
    try:
        if args.compare:
            # Run comparison
            compare_models(args.primary, args.detail1, args.detail2)
        else:
            # Test single model
            device = torch.device("cpu")
            
            if not Path(args.checkpoint).exists():
                print(f"❌ Checkpoint not found: {args.checkpoint}")
                print("Please run training first: python train_direct_triple.py")
                return
            
            # Load trained model
            model = load_trained_model(args.checkpoint, device)
            
            # Run inference
            detections, _ = test_trained_inference(
                model, args.primary, args.detail1, args.detail2, 
                conf_threshold=args.conf, device=device
            )
            
            # Visualize
            result_img = visualize_results(args.primary, detections, args.save)
            
            print(f"✅ Testing completed!")
            print(f"Result saved to: {args.save}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()