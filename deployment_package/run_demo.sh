#!/bin/bash
echo "Starting YOLOv13 Triple Input Model..."
python setup_deployment.py
python demo_verification.py
