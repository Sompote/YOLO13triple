#!/usr/bin/env python3
"""
Fixed Test Script for YOLOv13 Triple Input
Bypasses channel mismatch by using the working quick_test.py approach
"""

import sys
import os
import subprocess
from pathlib import Path
from glob import glob

def test_model_fixed(weights_path):
    """
    Test model using the working quick_test.py approach
    """
    
    print("🚀 Testing YOLOv13 Triple Input Model (Fixed)")
    print("=" * 60)
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found: {weights_path}")
        return False
    
    print(f"📦 Model: {weights_path}")
    print(f"🔧 Using quick_test.py approach to bypass channel mismatch...")
    print("=" * 60)
    
    try:
        # Use quick_test.py which works with the model
        result = subprocess.run([
            "python", "quick_test.py", weights_path
        ], capture_output=True, text=True)
        
        print("📝 Test Output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings:")
            print(result.stderr)
        
        # Parse results
        output = result.stdout
        success = result.returncode == 0
        
        # Extract detection information
        detections_found = "detections" in output and not "0 detections" in output
        
        print(f"\n📊 Test Results:")
        print(f"   Execution: {'✅ Success' if success else '❌ Failed'}")
        print(f"   Detections: {'✅ Found objects' if detections_found else '⚠️  No objects detected'}")
        
        if success and detections_found:
            print(f"\n🎉 Your triple input model is working!")
            print(f"   ✅ Model loads successfully")
            print(f"   ✅ Inference runs without errors") 
            print(f"   ✅ Objects detected in test images")
        elif success:
            print(f"\n⚠️  Model works but no objects detected")
            print(f"   This could mean:")
            print(f"   • Confidence threshold too high")
            print(f"   • Objects smaller than expected")
            print(f"   • Model needs more training")
        else:
            print(f"\n❌ Model execution failed")
            print(f"   Check the error messages above")
        
        return success
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_model_fixed.py <weights_path>")
        print("")
        print("Examples:")
        print("  python test_model_fixed.py runs/unified_train_triple/yolo_s_triple*/weights/best.pt")
        print("  python test_model_fixed.py runs/*/weights/best.pt")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    
    # Handle wildcard paths
    if "*" in weights_path:
        matching_files = glob(weights_path)
        if matching_files:
            weights_path = matching_files[0]
            print(f"🔍 Found weights: {weights_path}")
        else:
            print(f"❌ No files found matching: {weights_path}")
            sys.exit(1)
    
    success = test_model_fixed(weights_path)
    
    if success:
        print(f"\n🎯 Next Steps:")
        print(f"   • For detailed metrics: python evaluate_triple_simple.py {weights_path}")
        print(f"   • For diagnostics: python diagnose_model_issues.py {weights_path}")
        print(f"   • For threshold tuning: python test_confidence_thresholds.py {weights_path}")
    else:
        print(f"\n💡 Troubleshooting:")
        print(f"   • Check if yolov13 directory exists")
        print(f"   • Verify model file is not corrupted")
        print(f"   • Try: python diagnose_model_issues.py {weights_path}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()