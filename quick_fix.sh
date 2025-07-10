#!/bin/bash

# Quick Fix for YOLOv13 Triple Input Cloud Deployment
# One-command solution for common issues

echo "🚀 Quick Fix for YOLOv13 Cloud Issues"
echo "===================================="

# Fix NumPy version conflict (most common issue)
echo "🔧 Fixing NumPy version conflict..."
pip install "numpy<2.0" --force-reinstall --no-warn-script-location

# Ensure ultralytics is properly installed
echo "📦 Ensuring ultralytics installation..."
pip install ultralytics --force-reinstall --no-warn-script-location

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Set Python path
export PYTHONPATH="${PWD}:$PYTHONPATH"

# Make scripts executable
chmod +x standalone_train_fixed.py fix_cloud_issues.py

echo "✅ Quick fix complete!"
echo ""
echo "🎯 Now run:"
echo "  python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch 8 --device cpu"
echo ""
echo "💡 For GPU training:"
echo "  python3 standalone_train_fixed.py --data working_dataset.yaml --model s --epochs 50 --batch 16 --device 0"