#!/usr/bin/env python3
"""
Basic Usage Examples for YOLOv13 Triple Input

This script demonstrates how to use the triple input YOLO model
for object detection with 3 images.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from triple_inference import TripleYOLOModel, load_and_preprocess_images, visualize_results
import torch
import cv2
import numpy as np


def example_1_basic_inference():
    """Example 1: Basic triple image inference."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Triple Image Inference")
    print("=" * 60)
    
    # Create sample images (in practice, load your own images)
    def create_sample_image(text, color=(255, 255, 255)):
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        return img
    
    # Create and save sample images
    primary_img = create_sample_image("Primary Image", (0, 255, 0))
    detail1_img = create_sample_image("Detail 1", (255, 0, 0))
    detail2_img = create_sample_image("Detail 2", (0, 0, 255))
    
    cv2.imwrite("example_primary.jpg", primary_img)
    cv2.imwrite("example_detail1.jpg", detail1_img)
    cv2.imwrite("example_detail2.jpg", detail2_img)
    
    # Load model
    print("1. Loading triple input YOLO model...")
    model = TripleYOLOModel(nc=80)
    model.eval()
    
    # Load and preprocess images
    print("2. Loading and preprocessing images...")
    images = load_and_preprocess_images(
        "example_primary.jpg",
        "example_detail1.jpg", 
        "example_detail2.jpg"
    )
    
    # Run inference
    print("3. Running inference...")
    with torch.no_grad():
        batched_input = [img.unsqueeze(0) for img in images]
        predictions = model(batched_input)
    
    print(f"4. Results: {type(predictions)}")
    if isinstance(predictions, tuple):
        print(f"   Output shape: {predictions[0].shape}")
    
    # Visualize results
    print("5. Saving visualization...")
    result_img = visualize_results("example_primary.jpg", None, "example_result.jpg")
    
    print("✅ Example 1 completed! Check example_result.jpg")
    
    # Cleanup
    for f in ["example_primary.jpg", "example_detail1.jpg", "example_detail2.jpg"]:
        Path(f).unlink(missing_ok=True)


def example_2_custom_model():
    """Example 2: Using the model with custom settings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Model Configuration")
    print("=" * 60)
    
    # Create model with custom number of classes
    print("1. Creating model with custom classes...")
    model = TripleYOLOModel(nc=10)  # Only 10 classes instead of 80
    model.eval()
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test with random inputs
    print("2. Testing with random inputs...")
    dummy_images = [torch.randn(1, 3, 640, 640) for _ in range(3)]
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"3. Output format: {type(outputs)}")
    if isinstance(outputs, tuple) and len(outputs) >= 1:
        if isinstance(outputs[0], torch.Tensor):
            print(f"   Detection shape: {outputs[0].shape}")
            print(f"   Expected format: [batch, detections, attributes]")
            print(f"   Attributes: [x, y, w, h, conf, class_scores...]")
    
    print("✅ Example 2 completed!")


def example_3_batch_processing():
    """Example 3: Batch processing multiple image sets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    # Load model
    model = TripleYOLOModel(nc=80)
    model.eval()
    
    # Simulate multiple image sets
    print("1. Preparing batch data...")
    batch_size = 2
    
    # Create batch of triple images
    batch_images = []
    for i in range(batch_size):
        sample_set = [torch.randn(3, 640, 640) for _ in range(3)]
        batch_images.append(sample_set)
    
    print(f"2. Processing {batch_size} image sets...")
    results = []
    
    with torch.no_grad():
        for i, image_set in enumerate(batch_images):
            # Add batch dimension
            batched_set = [img.unsqueeze(0) for img in image_set]
            
            # Run inference
            output = model(batched_set)
            results.append(output)
            
            print(f"   Set {i+1}: Processed successfully")
    
    print(f"3. Batch processing completed! Processed {len(results)} sets")
    print("✅ Example 3 completed!")


def example_4_error_handling():
    """Example 4: Error handling and fallback scenarios."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Error Handling")
    print("=" * 60)
    
    from ultralytics.nn.modules.conv import TripleInputConv
    
    # Test TripleInputConv with different input scenarios
    print("1. Testing TripleInputConv fallback behavior...")
    conv = TripleInputConv(3, 64, 3, 2)
    
    # Scenario 1: Proper triple input
    print("   Scenario 1: Triple input")
    triple_input = [torch.randn(1, 3, 640, 640) for _ in range(3)]
    output1 = conv(triple_input)
    print(f"   ✅ Triple input shape: {output1.shape}")
    
    # Scenario 2: Single input (fallback)
    print("   Scenario 2: Single input fallback")
    single_input = torch.randn(1, 3, 640, 640)
    output2 = conv(single_input)
    print(f"   ✅ Single input shape: {output2.shape}")
    
    # Scenario 3: Wrong number of inputs
    print("   Scenario 3: Wrong input count")
    try:
        wrong_input = [torch.randn(1, 3, 640, 640) for _ in range(2)]  # Only 2 images
        output3 = conv(wrong_input)
        print(f"   ⚠️  Processed with fallback: {output3.shape}")
    except Exception as e:
        print(f"   ❌ Error handled: {str(e)[:50]}...")
    
    print("✅ Example 4 completed!")


def example_5_performance_comparison():
    """Example 5: Performance comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Performance Comparison")
    print("=" * 60)
    
    import time
    
    # Create models
    model = TripleYOLOModel(nc=80)
    model.eval()
    
    # Prepare test data
    triple_images = [torch.randn(1, 3, 640, 640) for _ in range(3)]
    single_image = torch.randn(1, 3, 640, 640)
    
    # Warm up
    print("1. Warming up models...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(triple_images)
    
    # Benchmark triple input
    print("2. Benchmarking triple input...")
    num_runs = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(triple_images)
    
    triple_time = (time.time() - start_time) / num_runs
    
    # Benchmark single input (using first image)
    print("3. Benchmarking single input fallback...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model([single_image])  # Pass as list for consistency
    
    single_time = (time.time() - start_time) / num_runs
    
    # Results
    print(f"4. Results (average over {num_runs} runs):")
    print(f"   Triple input: {triple_time*1000:.1f}ms")
    print(f"   Single input: {single_time*1000:.1f}ms")
    print(f"   Overhead: {((triple_time/single_time - 1)*100):.1f}%")
    
    print("✅ Example 5 completed!")


def main():
    """Run all examples."""
    print("🚀 YOLOv13 Triple Input - Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_inference()
        example_2_custom_model()
        example_3_batch_processing()
        example_4_error_handling()
        example_5_performance_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Try with your own images: python triple_inference.py --primary img1.jpg --detail1 img2.jpg --detail2 img3.jpg")
        print("2. Train your own model: python train_direct_triple.py --data-dir your_dataset")
        print("3. Check the documentation: README.md")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()