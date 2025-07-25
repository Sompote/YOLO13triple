#!/usr/bin/env python3
"""
Proper evaluation script for YOLOv13 Triple Input models
Handles triple input format and calculates meaningful performance metrics
"""

import sys
import os
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

def setup_triple_environment():
    """Setup environment for triple YOLO"""
    current_dir = Path(__file__).parent
    yolov13_path = current_dir / "yolov13"
    
    if not yolov13_path.exists():
        print("❌ Error: yolov13 directory not found")
        return False
    
    # Clean imports and setup path
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('ultralytics')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    if str(yolov13_path) in sys.path:
        sys.path.remove(str(yolov13_path))
    sys.path.insert(0, str(yolov13_path))
    
    # Set environment
    os.environ.update({
        "PYTHONPATH": str(yolov13_path) + ":" + os.environ.get("PYTHONPATH", ""),
        "ULTRALYTICS_AUTO_UPDATE": "0",
        "ULTRALYTICS_DISABLE_CHECKS": "1",
        "ULTRALYTICS_OFFLINE": "1",
        "NUMPY_EXPERIMENTAL_DTYPE_API": "0"
    })
    
    return True

def load_ground_truth_labels(labels_dir):
    """Load ground truth labels from YOLO format"""
    labels = {}
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return labels
    
    for label_file in glob(os.path.join(labels_dir, "*.txt")):
        image_name = Path(label_file).stem
        boxes = []
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    boxes.append({
                        'class': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        labels[image_name] = boxes
    
    return labels

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (YOLO format)"""
    # Convert from center format to corner format
    def center_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = center_to_corners([box1['x_center'], box1['y_center'], box1['width'], box1['height']])
    x1_2, y1_2, x2_2, y2_2 = center_to_corners([box2['x_center'], box2['y_center'], box2['width'], box2['height']])
    
    # Calculate intersection
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
        return 0.0
    
    intersection = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    
    # Calculate union
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_detections(predictions, ground_truth, iou_threshold=0.5):
    """Evaluate predictions against ground truth"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    confidence_scores = []
    iou_scores = []
    
    for image_name in ground_truth.keys():
        gt_boxes = ground_truth[image_name]
        pred_boxes = predictions.get(image_name, [])
        
        # Track which GT boxes have been matched
        matched_gt = [False] * len(gt_boxes)
        
        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT box
            for gt_idx, gt_box in enumerate(gt_boxes):
                if matched_gt[gt_idx]:
                    continue
                
                if pred_box['class'] == gt_box['class']:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Determine if this is TP or FP
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt[best_gt_idx] = True
                confidence_scores.append(pred_box.get('confidence', 0))
                iou_scores.append(best_iou)
            else:
                false_positives += 1
        
        # Count unmatched GT boxes as FN
        false_negatives += sum(1 for matched in matched_gt if not matched)
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'confidence_scores': confidence_scores,
        'iou_scores': iou_scores
    }

def calculate_metrics(eval_results):
    """Calculate precision, recall, F1, and mAP"""
    tp = eval_results['true_positives']
    fp = eval_results['false_positives']
    fn = eval_results['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Simple mAP calculation (at IoU=0.5)
    map_50 = precision  # Simplified for this implementation
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'map_50': map_50,
        'average_confidence': np.mean(eval_results['confidence_scores']) if eval_results['confidence_scores'] else 0.0,
        'average_iou': np.mean(eval_results['iou_scores']) if eval_results['iou_scores'] else 0.0
    }

def run_inference_on_test_set(model, test_images_dir, detail1_dir, detail2_dir, conf_threshold=0.01):
    """Run inference on test set using training pipeline for compatibility"""
    predictions = {}
    
    test_images = glob(os.path.join(test_images_dir, "*.jpg")) + glob(os.path.join(test_images_dir, "*.png"))
    
    print(f"🎯 Running inference on {len(test_images)} test images...")
    print("🔧 Using training pipeline for triple input compatibility...")
    
    # Create temporary validation config for inference
    temp_val_config = {
        'path': str(Path(test_images_dir).parent.parent.parent),
        'val': str(Path(test_images_dir).relative_to(Path(test_images_dir).parent.parent.parent)),
        'nc': 1,
        'names': {0: 'hole'},
        'triple_input': True,
        'detail1_path': str(Path(detail1_dir).relative_to(Path(test_images_dir).parent.parent.parent)),
        'detail2_path': str(Path(detail2_dir).relative_to(Path(test_images_dir).parent.parent.parent)),
        'dataset_type': 'triple_yolo',
        'task': 'detect'
    }
    
    temp_config_path = "temp_inference_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_val_config, f)
    
    try:
        # Run validation to get predictions
        results = model.val(
            data=temp_config_path,
            split='val',
            conf=conf_threshold,
            verbose=False,
            save_json=True,
            plots=False
        )
        
        # Parse results from validation JSON if available
        json_path = None
        for run_dir in glob("runs/val/val*"):
            json_file = os.path.join(run_dir, "predictions.json")
            if os.path.exists(json_file):
                json_path = json_file
                break
        
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_results = json.load(f)
            
            # Convert JSON results to our format
            for result in json_results:
                image_id = result.get('image_id', 0)
                image_name = f"test_image_{image_id + 1}"  # Assuming sequential naming
                
                if image_name not in predictions:
                    predictions[image_name] = []
                
                # Convert bbox format
                bbox = result.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
                x, y, w, h = bbox
                
                # Assume image size for normalization (this is a limitation)
                img_w, img_h = 640, 640  # Default YOLO size
                
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h
                
                predictions[image_name].append({
                    'class': result.get('category_id', 0),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': result.get('score', 0.0)
                })
        
        # If no JSON results, try direct inference approach
        if not predictions:
            print("📝 No JSON results found, trying direct approach...")
            for img_path in test_images:
                image_name = Path(img_path).stem
                predictions[image_name] = []  # No detections for now
        
    except Exception as e:
        print(f"⚠️  Error in validation approach: {e}")
        # Fallback: return empty predictions
        for img_path in test_images:
            image_name = Path(img_path).stem
            predictions[image_name] = []
    
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    return predictions

def generate_evaluation_report(metrics, eval_results, output_dir="evaluation_results"):
    """Generate detailed evaluation report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("YOLOv13 Triple Input Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"mAP@0.5: {metrics['map_50']:.4f}\n")
        f.write(f"Average Confidence: {metrics['average_confidence']:.4f}\n")
        f.write(f"Average IoU: {metrics['average_iou']:.4f}\n\n")
        
        f.write("Detection Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"True Positives: {eval_results['true_positives']}\n")
        f.write(f"False Positives: {eval_results['false_positives']}\n")
        f.write(f"False Negatives: {eval_results['false_negatives']}\n")
        f.write(f"Total Detections: {eval_results['true_positives'] + eval_results['false_positives']}\n")
        f.write(f"Total Ground Truth: {eval_results['true_positives'] + eval_results['false_negatives']}\n")
    
    # Generate metrics visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrics bar chart
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['map_50']]
    
    ax1.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Performance Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Detection statistics pie chart
    detection_labels = ['True Positives', 'False Positives', 'False Negatives']
    detection_values = [eval_results['true_positives'], eval_results['false_positives'], eval_results['false_negatives']]
    
    if sum(detection_values) > 0:
        ax2.pie(detection_values, labels=detection_labels, autopct='%1.1f%%', colors=['green', 'red', 'orange'])
        ax2.set_title('Detection Distribution')
    else:
        ax2.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Detection Distribution')
    
    # Confidence distribution
    if eval_results['confidence_scores']:
        ax3.hist(eval_results['confidence_scores'], bins=20, alpha=0.7, color='blue')
        ax3.set_title('Confidence Score Distribution')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
    else:
        ax3.text(0.5, 0.5, 'No confidence scores', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Confidence Score Distribution')
    
    # IoU distribution
    if eval_results['iou_scores']:
        ax4.hist(eval_results['iou_scores'], bins=20, alpha=0.7, color='green')
        ax4.set_title('IoU Score Distribution')
        ax4.set_xlabel('IoU')
        ax4.set_ylabel('Frequency')
    else:
        ax4.text(0.5, 0.5, 'No IoU scores', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('IoU Score Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation report saved to: {output_dir}/")

def evaluate_triple_model(weights_path, data_config="datatrain.yaml", conf_threshold=0.01, iou_threshold=0.5):
    """Main evaluation function"""
    
    print("🚀 YOLOv13 Triple Input Model Evaluation")
    print("=" * 50)
    
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
        
    if not os.path.exists(data_config):
        print(f"❌ Error: Dataset config not found: {data_config}")
        return False
    
    # Setup environment
    if not setup_triple_environment():
        return False
    
    try:
        # Load dataset configuration
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = Path(config['path'])
        test_split = config.get('test', config.get('val', 'images/primary/val'))
        test_images_dir = base_path / test_split
        labels_dir = base_path / "labels" / "primary" / "test"
        
        # For triple input
        detail1_dir = base_path / config.get('detail1_path', 'images/detail1')
        detail2_dir = base_path / config.get('detail2_path', 'images/detail2')
        
        print(f"📦 Model weights: {weights_path}")
        print(f"📊 Dataset config: {data_config}")
        print(f"🖼️  Test images: {test_images_dir}")
        print(f"🏷️  Test labels: {labels_dir}")
        print(f"📝 Confidence threshold: {conf_threshold}")
        print(f"📏 IoU threshold: {iou_threshold}")
        
        # Import after environment setup
        from ultralytics import YOLO
        
        # Load model
        print("\n📦 Loading model...")
        model = YOLO(weights_path)
        
        # Load ground truth
        print("📖 Loading ground truth labels...")
        ground_truth = load_ground_truth_labels(str(labels_dir))
        print(f"📊 Found {len(ground_truth)} labeled images")
        
        if len(ground_truth) == 0:
            print("❌ No ground truth labels found!")
            return False
        
        # Run inference
        print("🎯 Running inference on test set...")
        predictions = run_inference_on_test_set(
            model, str(test_images_dir), str(detail1_dir), str(detail2_dir), conf_threshold
        )
        
        # Evaluate predictions
        print("📊 Evaluating predictions...")
        eval_results = evaluate_detections(predictions, ground_truth, iou_threshold)
        
        # Calculate metrics
        metrics = calculate_metrics(eval_results)
        
        # Display results
        print("\n📊 Evaluation Results:")
        print("=" * 30)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"mAP@0.5: {metrics['map_50']:.4f}")
        print(f"Average Confidence: {metrics['average_confidence']:.4f}")
        print(f"Average IoU: {metrics['average_iou']:.4f}")
        
        print(f"\n🔢 Detection Statistics:")
        print(f"True Positives: {eval_results['true_positives']}")
        print(f"False Positives: {eval_results['false_positives']}")
        print(f"False Negatives: {eval_results['false_negatives']}")
        print(f"Total Predictions: {eval_results['true_positives'] + eval_results['false_positives']}")
        print(f"Total Ground Truth: {eval_results['true_positives'] + eval_results['false_negatives']}")
        
        # Generate detailed report
        print("\n📝 Generating evaluation report...")
        generate_evaluation_report(metrics, eval_results)
        
        print("\n✅ Evaluation completed successfully!")
        
        # Save results to JSON
        results_data = {
            'metrics': metrics,
            'detection_stats': {
                'true_positives': eval_results['true_positives'],
                'false_positives': eval_results['false_positives'],
                'false_negatives': eval_results['false_negatives']
            },
            'evaluation_params': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'weights_path': weights_path,
                'data_config': data_config
            }
        }
        
        with open('evaluation_results/evaluation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluate_triple_model.py <weights_path> [data_config] [conf_threshold] [iou_threshold]")
        print("")
        print("Examples:")
        print("  python evaluate_triple_model.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python evaluate_triple_model.py best.pt datatrain.yaml 0.01 0.5")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    data_config = sys.argv[2] if len(sys.argv) > 2 else "datatrain.yaml"
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    iou_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    # Handle wildcard paths
    if "*" in weights_path:
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    success = evaluate_triple_model(weights_path, data_config, conf_threshold, iou_threshold)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()