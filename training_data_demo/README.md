# Triple Input YOLO Training Data

## Directory Structure
- `images/primary/`: Primary images with object labels
- `images/detail1/`: First set of detail images
- `images/detail2/`: Second set of detail images  
- `labels/`: YOLO format label files (corresponding to primary images)

## Data Preparation
1. Place your primary images (with objects to detect) in `images/primary/train/` and `images/primary/val/`
2. Place corresponding detail images in `images/detail1/` and `images/detail2/` directories
3. Place YOLO format labels in `labels/train/` and `labels/val/`

## Label Format
Each label file should have the same name as the primary image but with .txt extension.
Label format: `class_id center_x center_y width height` (normalized 0-1)

## Training
Run: `python train_triple.py --data triple_dataset.yaml --model yolov13-triple.yaml`
