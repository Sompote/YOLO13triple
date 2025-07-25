#!/usr/bin/env python3
"""
Simple evaluation script that uses the working validation pipeline
and extracts metrics for proper evaluation reporting
"""

import sys
import os
import yaml
import json
import subprocess
import re
from pathlib import Path
from glob import glob

def run_validation_and_extract_metrics(weights_path, data_config="datatrain.yaml"):
    """Run validation using working pipeline and extract metrics"""
    
    print("🚀 YOLOv13 Triple Input Model Evaluation")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
        
    if not os.path.exists(data_config):
        print(f"❌ Error: Dataset config not found: {data_config}")
        return False
    
    print(f"📦 Model weights: {weights_path}")
    print(f"📊 Dataset config: {data_config}")
    
    # Load dataset configuration to show test info
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = Path(config['path'])
        test_split = config.get('test', config.get('val', 'images/primary/val'))
        test_images_dir = base_path / test_split
        labels_dir = base_path / "labels" / "primary" / "test"
        
        print(f"🖼️  Test images: {test_images_dir}")
        print(f"🏷️  Test labels: {labels_dir}")
        
        # Count test images and labels
        test_images = glob(str(test_images_dir / "*.jpg")) + glob(str(test_images_dir / "*.png"))
        label_files = glob(str(labels_dir / "*.txt"))
        print(f"📊 Found {len(test_images)} test images and {len(label_files)} label files")
        
    except Exception as e:
        print(f"⚠️  Could not read dataset config: {e}")
    
    print("\n🎯 Running validation using working pipeline...")
    print("=" * 50)
    
    # Run the working validation script and capture output
    try:
        result = subprocess.run([
            "python", "run_validation.py", weights_path, data_config
        ], capture_output=True, text=True)
        
        print("📝 Validation Output:")
        print("-" * 30)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Validation Warnings/Errors:")
            print(result.stderr)
        
        # Parse metrics from output
        metrics = parse_validation_output(result.stdout)
        
        # Display extracted metrics
        print("\n📊 Extracted Evaluation Metrics:")
        print("=" * 40)
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        else:
            print("❌ No metrics could be extracted from validation output")
        
        # Generate simple report
        generate_simple_report(metrics, weights_path, data_config)
        
        print("\n✅ Evaluation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

def parse_validation_output(output):
    """Parse metrics from validation output"""
    metrics = {}
    
    # Look for common YOLO validation metrics
    patterns = {
        'mAP@0.5': r'mAP@0\.5:\s*([0-9.]+)',
        'mAP@0.5:0.95': r'mAP@0\.5:0\.95:\s*([0-9.]+)',
        'Precision': r'Precision:\s*([0-9.]+)',
        'Recall': r'Recall:\s*([0-9.]+)',
        'F1-Score': r'F1-Score:\s*([0-9.]+)',
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                metrics[metric_name] = 0.0
    
    # Look for speed metrics
    speed_patterns = {
        'Preprocessing': r'Preprocessing:\s*([0-9.]+)\s*ms',
        'Inference': r'Inference:\s*([0-9.]+)\s*ms',
        'Postprocessing': r'Postprocessing:\s*([0-9.]+)\s*ms',
        'Total': r'Total:\s*([0-9.]+)\s*ms'
    }
    
    for metric_name, pattern in speed_patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                metrics[f'Speed_{metric_name}'] = float(match.group(1))
            except ValueError:
                metrics[f'Speed_{metric_name}'] = 0.0
    
    # If no metrics found, check for validation success indicators
    if not metrics:
        if "validation completed successfully" in output.lower():
            metrics['Status'] = 'Completed'
        else:
            metrics['Status'] = 'Failed'
    
    return metrics

def generate_simple_report(metrics, weights_path, data_config, output_dir="evaluation_results"):
    """Generate simple evaluation report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text report
    report_path = os.path.join(output_dir, "simple_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("YOLOv13 Triple Input Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {weights_path}\n")
        f.write(f"Dataset: {data_config}\n")
        f.write(f"Evaluation Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        else:
            f.write("No metrics available\n")
        
        f.write(f"\nNote: This evaluation uses the triple input validation pipeline\n")
        f.write(f"which properly handles 9-channel (3x RGB) input format.\n")
    
    # Save metrics to JSON
    metrics_json_path = os.path.join(output_dir, "simple_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'model_path': weights_path,
            'data_config': data_config,
            'evaluation_type': 'triple_input_validation'
        }, f, indent=2)
    
    print(f"📊 Simple evaluation report saved to: {output_dir}/")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluate_triple_simple.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python evaluate_triple_simple.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python evaluate_triple_simple.py best.pt datatrain.yaml")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    
    # Handle wildcard paths
    if "*" in weights_path:
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    success = run_validation_and_extract_metrics(weights_path, data_config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()