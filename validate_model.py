#!/usr/bin/env python3
"""
Simple validation script using the training pipeline
This ensures compatibility with triple input models
"""

import sys
import os
from pathlib import Path

def validate_model():
    """Run validation using the training pipeline"""
    
    if len(sys.argv) < 2:
        print("Usage: python validate_model.py <weights_path> [data_config]")
        print("Example: python validate_model.py runs/unified_train_triple/yolo_s_triple49/weights/best.pt")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    
    # Handle wildcard paths
    if "*" in weights_path:
        from glob import glob
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    print("🚀 Running validation using training pipeline...")
    print(f"📦 Model: {weights_path}")
    print(f"📊 Data: {data_config}")
    
    # Use the unified training script with validation task
    cmd = f"python unified_train_optimized.py --data {data_config} --task val --model {weights_path} --split test"
    
    print(f"🎯 Executing: {cmd}")
    print("=" * 60)
    
    # Execute the command
    os.system(cmd)

if __name__ == "__main__":
    validate_model()