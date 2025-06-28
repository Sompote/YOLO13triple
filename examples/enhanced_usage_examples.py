#!/usr/bin/env python3
"""
Enhanced Usage Examples for YOLOv13 Triple Input with Pretrained Weights
Demonstrates model variants, pretrained weights, and transfer learning
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.triple_yolo_variants import (
    create_triple_yolo_model, MODEL_VARIANTS,
    triple_yolo13n, triple_yolo13s, triple_yolo13m, triple_yolo13l, triple_yolo13x
)
from enhanced_triple_inference import EnhancedTripleInference


def example_1_model_variants():
    """Example 1: Demonstrate different model variants."""
    print("=" * 60)
    print("EXAMPLE 1: Model Variants Comparison")
    print("=" * 60)
    
    variants = ['n', 's', 'm']  # Test smaller variants for demo
    models = {}
    
    for variant in variants:
        print(f"\n🧪 Creating YOLOv13{variant} Triple Input Model:")
        
        # Create model
        model = create_triple_yolo_model(variant=variant, nc=80)
        models[variant] = model
        
        # Print model info
        info = model.get_model_info()
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Depth Multiple: {info['depth_multiple']}")
        print(f"   Width Multiple: {info['width_multiple']}")
        
        # Test inference
        dummy_images = [torch.randn(1, 3, 640, 640) for _ in range(3)]
        with torch.no_grad():
            output = model(dummy_images)
        
        if isinstance(output, (tuple, list)):
            print(f"   Output shapes: {[item.shape for item in output if hasattr(item, 'shape')]}")
        else:
            print(f"   Output shape: {output.shape}")
        print(f"   ✅ YOLOv13{variant} working!")
    
    # Compare model sizes
    print(f"\n📊 Model Size Comparison:")
    for variant in variants:
        info = models[variant].get_model_info()
        print(f"   YOLOv13{variant}: {info['total_parameters']:,} parameters")


def example_2_pretrained_weights():
    """Example 2: Loading pretrained weights (simulated)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Pretrained Weights Loading")
    print("=" * 60)
    
    # Create model without pretrained weights
    print("1. Creating model without pretrained weights...")
    model_random = create_triple_yolo_model(variant='n', nc=80, pretrained=None)
    
    # Get initial weights for comparison
    initial_conv_weight = model_random.conv1.conv.weight.clone()
    
    print(f"   Initial conv1 weight mean: {initial_conv_weight.mean().item():.6f}")
    print(f"   Initial conv1 weight std: {initial_conv_weight.std().item():.6f}")
    
    # Simulate loading pretrained weights (create fake checkpoint)
    print("\n2. Creating simulated pretrained checkpoint...")
    fake_checkpoint = {
        'model_state_dict': model_random.state_dict(),
        'epoch': 50,
        'train_loss': 0.123,
        'val_loss': 0.098
    }
    
    # Save fake checkpoint
    checkpoint_path = "fake_pretrained.pt"
    torch.save(fake_checkpoint, checkpoint_path)
    
    print("3. Loading model with 'pretrained' weights...")
    try:
        model_pretrained = create_triple_yolo_model(
            variant='n', 
            nc=80, 
            pretrained=checkpoint_path
        )
        print("   ✅ Pretrained weights loaded successfully!")
        
        # Compare weights
        new_conv_weight = model_pretrained.conv1.conv.weight
        print(f"   Loaded conv1 weight mean: {new_conv_weight.mean().item():.6f}")
        print(f"   Weights are identical: {torch.equal(initial_conv_weight, new_conv_weight)}")
        
    except Exception as e:
        print(f"   ❌ Error loading pretrained weights: {e}")
    
    # Cleanup
    Path(checkpoint_path).unlink(missing_ok=True)


def example_3_transfer_learning():
    """Example 3: Transfer learning with frozen backbone."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Transfer Learning")
    print("=" * 60)
    
    print("1. Creating model with frozen backbone...")
    model = create_triple_yolo_model(
        variant='s',
        nc=10,  # Custom number of classes
        pretrained=None,  # Would use actual pretrained weights
        freeze_backbone=True
    )
    
    # Check which parameters are frozen
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"2. Parameter Analysis:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params*100:.1f}%")
    
    # Show which layers are frozen
    print(f"\n3. Layer Status:")
    for name, param in model.named_parameters():
        if 'conv' in name and '.weight' in name:
            status = "🔓 Trainable" if param.requires_grad else "🔒 Frozen"
            print(f"   {name}: {status}")
    
    # Demonstrate unfreezing
    print(f"\n4. Unfreezing backbone...")
    model.unfreeze_backbone()
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters after unfreezing: {trainable_after:,}")
    print(f"   ✅ All parameters now trainable!")


def example_4_enhanced_inference():
    """Example 4: Enhanced inference with different variants."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Enhanced Inference")
    print("=" * 60)
    
    # Create sample images for testing
    import cv2
    
    print("1. Creating sample images...")
    sample_dir = Path("temp_samples")
    sample_dir.mkdir(exist_ok=True)
    
    for i, name in enumerate(['primary', 'detail1', 'detail2']):
        # Create sample image
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"{name.title()} Image", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(sample_dir / f"{name}.jpg"), img)
    
    # Test different variants
    variants_to_test = ['n', 's']
    
    for variant in variants_to_test:
        print(f"\n2. Testing YOLOv13{variant} Enhanced Inference...")
        
        try:
            # Create inference object
            inference = EnhancedTripleInference(
                variant=variant,
                weights=None,  # No pretrained weights for demo
                device='cpu',
                nc=80,
                conf_thresh=0.25
            )
            
            # Run inference
            results = inference.run_inference(
                str(sample_dir / "primary.jpg"),
                str(sample_dir / "detail1.jpg"),
                str(sample_dir / "detail2.jpg"),
                imgsz=640,
                save_path=str(sample_dir / f"result_{variant}.jpg")
            )
            
            detections, predictions, inference_time = results[:3]
            
            print(f"   ✅ YOLOv13{variant} inference completed!")
            print(f"   Inference time: {inference_time*1000:.1f}ms")
            print(f"   Detections: {len(detections)}")
            
            # Benchmark if first variant
            if variant == 'n':
                print(f"\n3. Benchmarking YOLOv13{variant}...")
                benchmark_results = inference.benchmark_model(
                    warmup_runs=3, benchmark_runs=10
                )
            
        except Exception as e:
            print(f"   ❌ Error with YOLOv13{variant}: {e}")
    
    # Cleanup
    import shutil
    shutil.rmtree(sample_dir)
    print(f"\n   🧹 Cleaned up temporary files")


def example_5_convenience_functions():
    """Example 5: Using convenience functions for different variants."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Convenience Functions")
    print("=" * 60)
    
    # Test convenience functions
    convenience_functions = [
        ('YOLOv13n', triple_yolo13n),
        ('YOLOv13s', triple_yolo13s),
        ('YOLOv13m', triple_yolo13m),
    ]
    
    for name, func in convenience_functions:
        print(f"\n🧪 Testing {name} convenience function:")
        
        try:
            # Create model using convenience function
            model = func(nc=20, pretrained=None, freeze_backbone=False)
            
            # Get model info
            info = model.get_model_info()
            print(f"   ✅ {name} created successfully!")
            print(f"   Parameters: {info['total_parameters']:,}")
            print(f"   Variant: {info['variant']}")
            
            # Test forward pass
            dummy_images = [torch.randn(1, 3, 320, 320) for _ in range(3)]
            with torch.no_grad():
                output = model(dummy_images)
            
            if isinstance(output, (tuple, list)):
                print(f"   Forward pass successful: {[item.shape for item in output if hasattr(item, 'shape')]}")
            else:
                print(f"   Forward pass successful: {output.shape}")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")


def example_6_training_configuration():
    """Example 6: Training configuration examples."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Training Configurations")
    print("=" * 60)
    
    # Example training configurations
    training_configs = {
        'Basic Training': {
            'variant': 'n',
            'pretrained': None,
            'freeze_backbone': False,
            'lr': 0.001,
            'epochs': 100,
            'description': 'Standard training from scratch'
        },
        'Transfer Learning': {
            'variant': 's',
            'pretrained': 'yolov13s',  # Would download pretrained weights
            'freeze_backbone': True,
            'lr': 0.0001,
            'epochs': 50,
            'description': 'Fine-tuning with frozen backbone'
        },
        'Full Fine-tuning': {
            'variant': 'm',
            'pretrained': 'yolov13m',
            'freeze_backbone': False,
            'lr': 0.00001,  # Lower LR for pretrained model
            'epochs': 200,
            'description': 'Full model fine-tuning'
        },
        'Large Model Training': {
            'variant': 'l',
            'pretrained': None,
            'freeze_backbone': False,
            'lr': 0.0005,
            'epochs': 300,
            'description': 'Large model from scratch'
        }
    }
    
    print("🎯 Recommended Training Configurations:")
    
    for config_name, config in training_configs.items():
        print(f"\n📋 {config_name}:")
        print(f"   Variant: YOLOv13{config['variant']}")
        print(f"   Pretrained: {config['pretrained'] or 'None'}")
        print(f"   Freeze Backbone: {config['freeze_backbone']}")
        print(f"   Learning Rate: {config['lr']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Description: {config['description']}")
        
        # Show command example
        freeze_flag = "--freeze-backbone" if config['freeze_backbone'] else ""
        pretrained_flag = f"--pretrained {config['pretrained']}" if config['pretrained'] else ""
        
        print(f"   Command: python enhanced_triple_training.py \\")
        print(f"            --variant {config['variant']} \\")
        if pretrained_flag:
            print(f"            {pretrained_flag} \\")
        if freeze_flag:
            print(f"            {freeze_flag} \\")
        print(f"            --lr {config['lr']} \\")
        print(f"            --epochs {config['epochs']} \\")
        print(f"            --data-dir your_dataset")


def main():
    """Run all enhanced examples."""
    print("🚀 YOLOv13 Triple Input - Enhanced Usage Examples")
    print("=" * 60)
    print("Demonstrating model variants, pretrained weights, and transfer learning")
    
    try:
        example_1_model_variants()
        example_2_pretrained_weights()
        example_3_transfer_learning()
        example_4_enhanced_inference()
        example_5_convenience_functions()
        example_6_training_configuration()
        
        print("\n" + "=" * 60)
        print("🎉 All enhanced examples completed successfully!")
        print("=" * 60)
        
        print("\nKey Features Demonstrated:")
        print("✅ Multiple model variants (n, s, m, l, x)")
        print("✅ Pretrained weight loading and compatibility")
        print("✅ Transfer learning with backbone freezing")
        print("✅ Enhanced inference with variant selection")
        print("✅ Convenience functions for easy model creation")
        print("✅ Training configuration recommendations")
        
        print(f"\nNext Steps:")
        print("1. Train with pretrained weights:")
        print("   python enhanced_triple_training.py --variant s --pretrained yolov13s --data-dir your_dataset")
        print("2. Fine-tune with frozen backbone:")
        print("   python enhanced_triple_training.py --variant m --pretrained yolov13m --freeze-backbone --data-dir your_dataset")
        print("3. Run enhanced inference:")
        print("   python enhanced_triple_inference.py --variant l --weights best.pt --primary img1.jpg --detail1 img2.jpg --detail2 img3.jpg")
        
    except Exception as e:
        print(f"\n❌ Error in enhanced examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()