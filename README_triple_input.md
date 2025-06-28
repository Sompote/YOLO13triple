# YOLOv13 Triple Input Implementation

This repository contains a modified version of YOLOv13 that processes **3 images simultaneously** instead of a single image:

1. **Primary Image**: Contains the objects to be detected (with labels)
2. **Detail Image 1**: Additional detail information for enhanced detection
3. **Detail Image 2**: Additional detail information for enhanced detection

## Features

- ✅ **Triple Input Architecture**: Modified YOLO13 to process 3 images
- ✅ **Attention Mechanism**: Intelligent fusion of multi-image features
- ✅ **Custom Data Pipeline**: Specialized data loading for triple images
- ✅ **Training Support**: Complete training pipeline for triple input
- ✅ **Detection Scripts**: Ready-to-use detection with triple images
- ✅ **Backward Compatibility**: Fallback to single image if needed

## Architecture Overview

### TripleInputConv Module
```python
class TripleInputConv(nn.Module):
    """
    Triple input convolution for processing 3 images simultaneously.
    
    Features:
    - Individual convolution for each input
    - Feature fusion with attention mechanism
    - Adaptive feature weighting
    """
```

### Model Configuration
- **Base Model**: YOLOv13 architecture
- **Input Processing**: TripleInputConv as first layer
- **Feature Fusion**: Attention-based feature combination
- **Output**: Standard YOLO detection format

## Quick Start

### 1. Test Implementation
```bash
# Test all components
python test_triple_implementation.py

# Create sample images for testing
python detect_triple.py --create-samples
```

### 2. Detection with Triple Images
```bash
# Detect objects using 3 images
python detect_triple.py \
    --primary sample_data/primary/image_1.jpg \
    --detail1 sample_data/detail1/image_1.jpg \
    --detail2 sample_data/detail2/image_1.jpg \
    --save result.jpg --show
```

### 3. Prepare Training Data
```bash
# Setup directory structure
python train_triple.py --setup-dirs --data-dir my_dataset

# Validate dataset
python train_triple.py --validate --data-dir my_dataset

# Create configuration file
python train_triple.py --create-config --data-dir my_dataset
```

### 4. Training
```bash
# Train the model
python train_triple.py \
    --data triple_dataset.yaml \
    --model yolov13-triple \
    --epochs 100 \
    --batch 16
```

## Dataset Structure

```
training_data/
├── images/
│   ├── primary/        # Primary images with labels
│   │   ├── train/
│   │   └── val/
│   ├── detail1/        # First detail images
│   │   ├── train/
│   │   └── val/
│   └── detail2/        # Second detail images
│       ├── train/
│       └── val/
└── labels/             # YOLO format labels (for primary images)
    ├── train/
    └── val/
```

## Data Preparation Guidelines

### Image Requirements
- **Same Dimensions**: All 3 images should have the same resolution
- **Synchronized Content**: Images should show the same scene/objects
- **Labels**: Only primary images need corresponding label files

### Label Format
Standard YOLO format (normalized coordinates):
```
class_id center_x center_y width height
```

### Example Dataset
```bash
# Primary image: scene.jpg (has objects to detect)
# Detail1 image: scene_enhanced.jpg (enhanced contrast/brightness)
# Detail2 image: scene_hsv.jpg (different color space)
# Label file: scene.txt (YOLO format labels)
```

## Advanced Usage

### Custom Model Training
```python
from ultralytics import YOLO

# Load triple input model
model = YOLO('yolov13-triple.yaml')

# Train with custom parameters
results = model.train(
    data='my_triple_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='auto'
)
```

### Custom Dataset Class
```python
from ultralytics.data.triple_dataset import TripleYOLODataset

# Create custom dataset
dataset = TripleYOLODataset(
    img_path='path/to/primary/images',
    imgsz=640,
    augment=True
)
```

## File Structure

```
yolo_3dual_input/
├── yolov13/                           # Modified YOLOv13 repository
│   └── ultralytics/
│       ├── cfg/models/v13/
│       │   └── yolov13-triple.yaml    # Triple input model config
│       ├── nn/modules/
│       │   └── conv.py                # TripleInputConv module
│       └── data/
│           └── triple_dataset.py      # Triple dataset loader
├── detect_triple.py                   # Triple image detection script
├── train_triple.py                    # Triple input training script
├── test_triple_implementation.py      # Test suite
└── README_triple_input.md            # This file
```

## Key Modifications

### 1. TripleInputConv Module (`conv.py`)
- Processes 3 input images simultaneously
- Individual convolution paths for each image
- Attention-based feature fusion
- Fallback to single image processing

### 2. Model Configuration (`yolov13-triple.yaml`)
- Uses TripleInputConv as first layer
- Maintains YOLO13 backbone architecture
- Standard detection head

### 3. Data Pipeline (`triple_dataset.py`)
- TripleYOLODataset class for loading 3 images
- Automatic image matching and resizing
- Custom transforms for triple inputs
- Fallback handling for missing images

### 4. Training Pipeline (`train_triple.py`)
- Complete training setup for triple inputs
- Dataset validation and configuration
- Custom hyperparameters for multi-image training

### 5. Detection Pipeline (`detect_triple.py`)
- Load and preprocess 3 images
- Run inference with triple input model
- Visualize results with multi-image info

## Technical Details

### Feature Fusion Strategy
1. **Individual Processing**: Each image processed by separate conv layers
2. **Feature Concatenation**: Combine features along channel dimension
3. **Fusion Layer**: 1x1 convolution to merge features
4. **Attention Weighting**: Channel attention for adaptive feature selection

### Memory Optimization
- Efficient tensor operations for triple inputs
- Batch processing support
- GPU memory management

### Backward Compatibility
- Automatic fallback to single image if only one provided
- Compatible with existing YOLO workflows
- Standard output format

## Performance Considerations

### Training
- **Memory**: Requires ~3x memory compared to single image
- **Speed**: Slightly slower due to additional processing
- **Convergence**: May require more epochs due to increased complexity

### Inference
- **Latency**: Minimal overhead with optimized fusion
- **Accuracy**: Potential improvement with complementary image information
- **Robustness**: Better performance in challenging conditions

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure yolov13 path is correct
   export PYTHONPATH="${PYTHONPATH}:/path/to/yolo_3dual_input/yolov13"
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python train_triple.py --batch 8
   ```

3. **Missing Images**
   - Check dataset structure
   - Ensure corresponding files exist
   - Use validation script to verify

### Debug Mode
```bash
# Enable verbose logging
python detect_triple.py --primary img1.jpg --detail1 img2.jpg --detail2 img3.jpg -v
```

## Future Enhancements

- [ ] Adaptive image weighting based on content
- [ ] Support for variable number of input images
- [ ] Temporal consistency for video inputs
- [ ] Integration with YOLO11/YOLO12 architectures
- [ ] Automated image selection and pairing
- [ ] Performance optimization for edge devices

## Contributing

1. Fork the repository
2. Create feature branch
3. Test with `test_triple_implementation.py`
4. Submit pull request

## License

Same as original YOLOv13 repository (AGPL-3.0)

## Citation

If you use this implementation, please cite:
```bibtex
@misc{yolov13_triple_input,
  title={YOLOv13 Triple Input Implementation},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/yolo_3dual_input}}
}
```