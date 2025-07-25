#!/usr/bin/env python3
"""
Simple Test Script for YOLOv13 Triple Input
Uses the same approach as the working run_validation.py script
"""

import sys
import os
import subprocess
from pathlib import Path
from glob import glob

def test_model_simple(weights_path, data_config="datatrain.yaml"):
    """
    Test trained model using the working validation pipeline
    """
    
    print("🚀 Testing YOLOv13 Triple Input Model")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
        
    if not os.path.exists(data_config):
        print(f"❌ Error: Dataset config not found: {data_config}")
        return False
    
    print(f"📦 Model: {weights_path}")
    print(f"📊 Data: {data_config}")
    print(f"\n🎯 Running test using working validation pipeline...")
    print("=" * 50)
    
    # Use the working run_validation.py script
    try:
        result = subprocess.run([
            "python", "run_validation.py", weights_path, data_config
        ], capture_output=True, text=True)
        
        print("📝 Test Output:")
        print("-" * 30)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Test Warnings/Errors:")
            print(result.stderr)
        
        # Parse and display key metrics
        output = result.stdout
        success = result.returncode == 0 and "validation completed successfully" in output.lower()
        
        print(f"\n📊 Test Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Try to extract metrics
        metrics = extract_metrics_from_output(output)
        if metrics:
            print(f"\n📈 Extracted Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def extract_metrics_from_output(output):
    """Extract metrics from validation output"""
    import re
    
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
    
    return metrics

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_model_simple.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python test_model_simple.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python test_model_simple.py best.pt datatrain.yaml")
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
    
    success = test_model_simple(weights_path, data_config)
    
    if not success:
        print(f"\n💡 Alternative: Try the comprehensive evaluation tools:")
        print(f"   python evaluate_triple_simple.py {weights_path}")
        print(f"   python diagnose_model_issues.py {weights_path}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()