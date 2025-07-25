#!/usr/bin/env python3
"""
Test different confidence thresholds to find optimal settings for small object detection
Uses the training pipeline for inference with varying confidence values
"""

import sys
import os
import subprocess
from pathlib import Path
from glob import glob

def test_confidence_thresholds(weights_path, data_config="datatrain.yaml"):
    """Test multiple confidence thresholds"""
    
    print("🎯 Testing Different Confidence Thresholds")
    print("=" * 60)
    
    if not os.path.exists(weights_path):
        print(f"❌ Model weights not found: {weights_path}")
        return False
    
    print(f"📦 Model: {weights_path}")
    print(f"📊 Data: {data_config}")
    
    # Test different confidence thresholds
    confidence_thresholds = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    
    print(f"\n🔬 Testing {len(confidence_thresholds)} confidence thresholds...")
    print("=" * 60)
    
    results = []
    
    for conf in confidence_thresholds:
        print(f"\n🎯 Testing confidence threshold: {conf}")
        print("-" * 40)
        
        try:
            # Use quick_test.py for inference with different confidence
            result = subprocess.run([
                "python", "quick_test.py", weights_path
            ], capture_output=True, text=True, env={
                **os.environ,
                "YOLO_CONF_THRESHOLD": str(conf)
            })
            
            # Parse output for detection counts
            output = result.stdout
            detections = parse_detection_counts(output)
            
            result_entry = {
                'confidence': conf,
                'detections': detections,
                'success': result.returncode == 0
            }
            results.append(result_entry)
            
            if detections:
                print(f"✅ Found {sum(detections.values())} total detections")
                for img, count in detections.items():
                    if count > 0:
                        print(f"   {img}: {count} detections")
            else:
                print("⚪ No detections found")
                
            if result.stderr:
                print(f"⚠️  Warnings: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"❌ Error at confidence {conf}: {e}")
            results.append({
                'confidence': conf,
                'detections': {},
                'success': False,
                'error': str(e)
            })
    
    # Generate summary
    print(f"\n📊 Confidence Threshold Test Results Summary")
    print("=" * 60)
    
    for result in results:
        conf = result['confidence']
        detections = result.get('detections', {})
        total_dets = sum(detections.values()) if detections else 0
        status = "✅ Success" if result['success'] else "❌ Failed"
        
        print(f"Confidence {conf:5.3f}: {total_dets:2d} detections - {status}")
    
    # Find optimal threshold
    successful_results = [r for r in results if r['success'] and r.get('detections')]
    if successful_results:
        # Find threshold with most detections
        best_result = max(successful_results, 
                         key=lambda x: sum(x['detections'].values()) if x['detections'] else 0)
        best_conf = best_result['confidence']
        best_count = sum(best_result['detections'].values())
        
        print(f"\n🏆 Optimal confidence threshold: {best_conf}")
        print(f"🎯 Total detections at optimal threshold: {best_count}")
        
        # Recommendation
        if best_count > 0:
            print(f"\n💡 Recommendation:")
            print(f"   Use confidence threshold: {best_conf}")
            print(f"   This setting detected objects on your test data")
        else:
            print(f"\n⚠️  No objects detected at any confidence level")
            print(f"   This might indicate:")
            print(f"   1. Model needs more training")
            print(f"   2. Test images don't contain target objects")
            print(f"   3. Objects are too small/difficult to detect")
    else:
        print(f"\n❌ No successful detections at any confidence threshold")
        print(f"   Model may need retraining or different evaluation approach")
    
    # Save results
    save_threshold_results(results, weights_path)
    
    return True

def parse_detection_counts(output):
    """Parse detection counts from quick_test output"""
    detections = {}
    
    lines = output.split('\n')
    for line in lines:
        if 'detections' in line and ':' in line:
            # Look for patterns like "Image 1: 3 detections"
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    img_part = parts[0].strip()
                    det_part = parts[1].strip()
                    
                    if 'detections' in det_part:
                        count_str = det_part.split()[0]
                        count = int(count_str)
                        detections[img_part] = count
            except (ValueError, IndexError):
                continue
    
    return detections

def save_threshold_results(results, weights_path, output_dir="evaluation_results"):
    """Save threshold test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    
    results_data = {
        'model_path': weights_path,
        'test_type': 'confidence_threshold_sweep',
        'results': results,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    results_path = os.path.join(output_dir, "confidence_threshold_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Also create text summary
    summary_path = os.path.join(output_dir, "confidence_threshold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Confidence Threshold Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {weights_path}\n")
        f.write(f"Test Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Results by Confidence Threshold:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            conf = result['confidence']
            detections = result.get('detections', {})
            total_dets = sum(detections.values()) if detections else 0
            status = "Success" if result['success'] else "Failed"
            
            f.write(f"Confidence {conf:5.3f}: {total_dets:2d} detections - {status}\n")
    
    print(f"📊 Threshold test results saved to: {output_dir}/")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_confidence_thresholds.py <weights_path> [data_config]")
        print("")
        print("Examples:")
        print("  python test_confidence_thresholds.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python test_confidence_thresholds.py best.pt datatrain.yaml")
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
    
    success = test_confidence_thresholds(weights_path, data_config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()