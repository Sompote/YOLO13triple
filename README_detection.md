# YOLOv13 Image Detection Guide

This repository contains scripts to perform object detection on images using YOLOv13 models with Hypergraph-based Adaptive Correlation Enhancement.

## Scripts Overview

### 1. `simple_detect.py` - Basic YOLOv13 Detection
The simplest way to detect objects in an image:

```python
from ultralytics import YOLO
import cv2

# Load YOLOv13 model and detect
model = YOLO("yolov13n.pt")
results = model("your_image.jpg")
annotated_image = results[0].plot()

# Display result
cv2.imshow("YOLOv13 Detection", annotated_image)
cv2.waitKey(0)
```

### 2. `detect_image.py` - Advanced YOLOv13 Detection
Full-featured detection script with command-line interface:

```bash
python detect_image.py --image path/to/image.jpg --model yolov13n.pt --show
```

### 3. `yolov13_detect.py` - Optimized YOLOv13 Script
Specialized script for YOLOv13 with batch processing:

```bash
python yolov13_detect.py --input image.jpg --model n --show --info
```

### 4. `yolov13_setup.py` - Installation Helper
Setup script to install dependencies and download models:

```bash
python yolov13_setup.py
```

## Usage Examples

### Quick Start
```bash
# Setup YOLOv13 environment (run once)
python yolov13_setup.py

# Simple detection with YOLOv13 Nano
python simple_detect.py

# Advanced detection with custom parameters
python yolov13_detect.py --input ultralytics/assets/bus.jpg --model s --conf 0.3 --show

# Batch process multiple images
python yolov13_detect.py --input /path/to/images/ --output /path/to/results/ --model l

# Get detailed detection information
python yolov13_detect.py --input image.jpg --info --model x
```

### Command Line Arguments

#### For `yolov13_detect.py` (recommended):
- `--input`: Input image path or directory (required)
- `--model`: YOLOv13 model size: n, s, l, x (default: n)
- `--output`: Output path for results (optional)
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.5)
- `--show`: Display the result image
- `--info`: Print detailed detection information

#### For `detect_image.py`:
- `--model`: Path to YOLOv13 model (.pt or .onnx) - default: yolov13n.pt
- `--image`: Path to input image (required)
- `--output`: Path to save output image (optional)
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.5)
- `--show`: Display the result image

## Model Options

### Pre-trained YOLOv13 Models
- `yolov13n.pt` - Nano (fastest, least accurate)
- `yolov13s.pt` - Small (balanced speed/accuracy)
- `yolov13l.pt` - Large (slower, higher accuracy)
- `yolov13x.pt` - Extra Large (slowest, most accurate)

**YOLOv13 Features:**
- Hypergraph-based Adaptive Correlation Enhancement
- Improved real-time object detection performance
- Better accuracy-speed trade-off compared to previous versions

### Model Download
YOLOv13 models are automatically downloaded on first use:

```python
from ultralytics import YOLO
model = YOLO('yolov13n.pt')  # Downloads automatically
```

Or use the setup script:
```bash
python yolov13_setup.py  # Downloads all YOLOv13 models
```

## Requirements

```bash
pip install ultralytics opencv-python
```

For ONNX support:
```bash
pip install onnxruntime
```

## Example Code Snippets

### Basic YOLOv13 Detection
```python
from ultralytics import YOLO

# Load YOLOv13 model
model = YOLO('yolov13n.pt')
results = model('image.jpg')
results[0].show()  # Display results
```

### Batch Processing with YOLOv13
```python
import glob
from ultralytics import YOLO

# Use larger model for batch processing
model = YOLO('yolov13l.pt')
images = glob.glob('images/*.jpg')

for img_path in images:
    results = model(img_path, conf=0.3)
    results[0].save(f'output/yolov13_{img_path.split("/")[-1]}')
```

### Using the Optimized Script
```python
from yolov13_detect import YOLOv13Detector

# Initialize detector
detector = YOLOv13Detector(model_size='l', confidence=0.3)

# Single image
detector.detect_single_image('image.jpg', save_path='result.jpg', show=True)

# Batch processing
detector.detect_batch('input_dir/', 'output_dir/')

# Get detailed info
info = detector.get_detection_info('image.jpg')
print(f"Found {info['num_detections']} objects")
```

### Custom Confidence and IoU
```python
# YOLOv13 with custom thresholds
model = YOLO('yolov13s.pt')
results = model('image.jpg', conf=0.3, iou=0.5)
```

### Access Detection Data
```python
results = model('image.jpg')
boxes = results[0].boxes
for box in boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xyxy}")
```

## Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## Performance Tips

1. **Choose appropriate YOLOv13 model size**:
   - `yolov13n`: Best for real-time applications, mobile devices
   - `yolov13s`: Balanced performance for most use cases
   - `yolov13l`: High accuracy for quality-critical applications
   - `yolov13x`: Maximum accuracy for research/analysis

2. **Optimize thresholds**:
   - Lower confidence (0.2-0.3) for detecting more objects
   - Higher confidence (0.6-0.8) for fewer false positives
   - Adjust IoU (0.3-0.7) based on object overlap tolerance

3. **Hardware optimization**:
   - GPU acceleration is automatic when CUDA is available
   - YOLOv13's hypergraph enhancement improves efficiency
   - Use batch processing for multiple images

4. **YOLOv13-specific benefits**:
   - Improved correlation enhancement reduces false positives
   - Better small object detection
   - More stable performance across different scenarios

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError: ultralytics**
   ```bash
   pip install ultralytics
   ```

2. **Model download fails**
   - Check internet connection
   - Try downloading manually from Ultralytics releases

3. **CUDA out of memory**
   - Use smaller model (yolov8n instead of yolov8x)
   - Reduce image size before detection

4. **Poor detection quality**
   - Lower confidence threshold (--conf 0.3)
   - Use larger YOLOv13 model (yolov13l or yolov13x)
   - Ensure good image quality and lighting
   - YOLOv13's hypergraph enhancement should improve results

5. **YOLOv13 setup issues**
   - Run `python yolov13_setup.py` to install dependencies
   - Ensure Python 3.8+ is installed
   - Check internet connection for model downloads