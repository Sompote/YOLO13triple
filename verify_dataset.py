#!/usr/bin/env python3
"""
Dataset verification script for YOLO training
Checks for common issues that cause zero performance metrics
"""

import os
from pathlib import Path
import yaml

def verify_dataset(dataset_yaml):
    """Verify YOLO dataset configuration and files"""
    print("🔍 YOLO Dataset Verification")
    print("=" * 50)
    
    # Load dataset config
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    train_path = dataset_path / config['train']
    val_path = dataset_path / config['val']
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"🚂 Train path: {train_path}")
    print(f"✅ Val path: {val_path}")
    print(f"🏷️ Classes: {config['nc']}")
    
    # Check paths exist
    issues = []
    
    if not dataset_path.exists():
        issues.append(f"❌ Dataset path doesn't exist: {dataset_path}")
    if not train_path.exists():
        issues.append(f"❌ Train images path doesn't exist: {train_path}")
    if not val_path.exists():
        issues.append(f"❌ Validation images path doesn't exist: {val_path}")
    
    if issues:
        for issue in issues:
            print(issue)
        return False
    
    # Get image and label files
    train_labels_path = dataset_path / "labels" / "train"
    val_labels_path = dataset_path / "labels" / "val"
    
    print(f"🏷️ Train labels: {train_labels_path}")
    print(f"🏷️ Val labels: {val_labels_path}")
    
    # Check train set
    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    train_labels = list(train_labels_path.glob("*.txt")) if train_labels_path.exists() else []
    
    print(f"\n📊 TRAINING SET:")
    print(f"   Images: {len(train_images)}")
    print(f"   Labels: {len(train_labels)}")
    
    if len(train_images) == 0:
        issues.append("❌ No training images found")
    if len(train_labels) == 0:
        issues.append("❌ No training labels found")
    
    # Check validation set
    val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
    val_labels = list(val_labels_path.glob("*.txt")) if val_labels_path.exists() else []
    
    print(f"\n📊 VALIDATION SET:")
    print(f"   Images: {len(val_images)}")
    print(f"   Labels: {len(val_labels)}")
    
    if len(val_images) == 0:
        issues.append("❌ No validation images found")
    if len(val_labels) == 0:
        issues.append("❌ No validation labels found")
    
    # Check image-label correspondence
    train_img_names = {img.stem for img in train_images}
    train_lbl_names = {lbl.stem for lbl in train_labels}
    val_img_names = {img.stem for img in val_images}
    val_lbl_names = {lbl.stem for lbl in val_labels}
    
    train_missing = train_img_names - train_lbl_names
    val_missing = val_img_names - val_lbl_names
    
    print(f"\n🔗 IMAGE-LABEL CORRESPONDENCE:")
    print(f"   Train matches: {len(train_img_names & train_lbl_names)}/{len(train_img_names)}")
    print(f"   Val matches: {len(val_img_names & val_lbl_names)}/{len(val_img_names)}")
    
    if train_missing:
        issues.append(f"❌ Train images missing labels: {list(train_missing)[:3]}...")
    if val_missing:
        issues.append(f"❌ Val images missing labels: {list(val_missing)[:3]}...")
    
    # Sample a few labels to check format
    print(f"\n🏷️ LABEL FORMAT CHECK:")
    sample_labels = train_labels[:2] + val_labels[:1]
    
    for label_file in sample_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        print(f"   {label_file.name}: {len(lines)} annotations")
        
        for i, line in enumerate(lines[:2]):  # Check first 2 annotations
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"❌ Wrong label format in {label_file.name} line {i+1}: expected 5 values, got {len(parts)}")
            else:
                try:
                    cls, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"❌ Coordinates not normalized in {label_file.name}: {x:.3f} {y:.3f} {w:.3f} {h:.3f}")
                    if cls >= config['nc']:
                        issues.append(f"❌ Class ID {cls} >= num_classes {config['nc']} in {label_file.name}")
                except ValueError:
                    issues.append(f"❌ Non-numeric values in {label_file.name} line {i+1}")
    
    # Check for cache files that might cause issues
    cache_files = list(dataset_path.glob("**/*.cache"))
    if cache_files:
        print(f"\n🗂️ CACHE FILES FOUND: {len(cache_files)}")
        print("   Consider removing cache files if changing dataset structure")
        for cache in cache_files[:3]:
            print(f"   - {cache}")
    
    # Summary
    print(f"\n📋 VERIFICATION SUMMARY:")
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Ensure every image has a corresponding .txt label file")
        print("   2. Check label format: class x_center y_center width height")
        print("   3. Verify coordinates are normalized (0-1 range)")
        print("   4. Remove .cache files: rm -rf dataset_path/**/*.cache")
        print("   5. Use different images for train/val sets")
        return False
    else:
        print("✅ Dataset verification passed!")
        print("📈 Ready for training")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify YOLO dataset")
    parser.add_argument("--data", type=str, default="working_dataset.yaml", 
                       help="Dataset YAML file")
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Dataset file not found: {args.data}")
        return
    
    success = verify_dataset(args.data)
    
    if success:
        print(f"\n🚀 Ready to train with: python simple_train.py --data {args.data}")
    else:
        print(f"\n🔧 Fix the issues above before training")

if __name__ == "__main__":
    main()