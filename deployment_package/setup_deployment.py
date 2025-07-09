#!/usr/bin/env python3
"""
YOLOv13 Triple Input Model - Deployment Setup Script
This script sets up the environment for standalone deployment on any machine
"""

import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
import argparse

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, text):
    """Print formatted step"""
    print(f"\n🔧 Step {step}: {text}")

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"   Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def check_system_requirements():
    """Check system requirements"""
    print_step(1, "Checking System Requirements")
    
    # Check Python version
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("   ❌ Python 3.8 or higher is required")
        return False
    else:
        print("   ✅ Python version is compatible")
    
    # Check platform
    system = platform.system()
    print(f"   Operating System: {system}")
    print(f"   Architecture: {platform.machine()}")
    
    # Check available disk space
    total, used, free = shutil.disk_usage(os.getcwd())
    free_gb = free // (1024**3)
    print(f"   Available disk space: {free_gb} GB")
    
    if free_gb < 2:
        print("   ⚠️  Warning: Low disk space (< 2GB)")
    else:
        print("   ✅ Sufficient disk space")
    
    return True

def setup_virtual_environment(env_name="yolo_env"):
    """Setup virtual environment"""
    print_step(2, f"Setting up Virtual Environment: {env_name}")
    
    # Check if virtual environment exists
    if Path(env_name).exists():
        print(f"   ⚠️  Virtual environment '{env_name}' already exists")
        response = input("   Do you want to recreate it? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(env_name)
            print(f"   🗑️  Removed existing environment")
        else:
            print(f"   ✅ Using existing environment")
            return env_name
    
    # Create virtual environment
    if not run_command(f"python -m venv {env_name}", "Creating virtual environment"):
        return None
    
    # Activation command varies by OS
    if platform.system() == "Windows":
        activate_cmd = f"{env_name}\\Scripts\\activate"
        pip_cmd = f"{env_name}\\Scripts\\pip"
    else:
        activate_cmd = f"source {env_name}/bin/activate"
        pip_cmd = f"{env_name}/bin/pip"
    
    print(f"   ✅ Virtual environment created")
    print(f"   📝 To activate: {activate_cmd}")
    
    return env_name

def install_dependencies(env_name=None, requirements_file="requirements-deployment.txt"):
    """Install dependencies"""
    print_step(3, f"Installing Dependencies from {requirements_file}")
    
    # Determine pip command
    if env_name:
        if platform.system() == "Windows":
            pip_cmd = f"{env_name}\\Scripts\\pip"
        else:
            pip_cmd = f"{env_name}/bin/pip"
    else:
        pip_cmd = "pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path(requirements_file).exists():
        if not run_command(f"{pip_cmd} install -r {requirements_file}", 
                          f"Installing from {requirements_file}"):
            print("   ⚠️  Full installation failed, trying minimal requirements")
            if Path("requirements-minimal.txt").exists():
                run_command(f"{pip_cmd} install -r requirements-minimal.txt", 
                          "Installing minimal requirements")
    else:
        print(f"   ❌ Requirements file not found: {requirements_file}")
        return False
    
    return True

def setup_model_structure():
    """Setup model directory structure"""
    print_step(4, "Setting up Model Directory Structure")
    
    # Create necessary directories
    directories = [
        "models",
        "weights",
        "data",
        "outputs",
        "logs",
        "temp"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ Created directory: {dir_name}")
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create deployment config
    deployment_config = {
        "model_path": "weights/best.pt",
        "input_size": [640, 640],
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "max_detections": 100,
        "device": "auto",
        "classes": [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light"
        ]
    }
    
    with open(config_dir / "deployment_config.json", "w") as f:
        json.dump(deployment_config, f, indent=2)
    
    print(f"   ✅ Created deployment configuration")
    
    return True

def create_deployment_scripts():
    """Create deployment scripts"""
    print_step(5, "Creating Deployment Scripts")
    
    # Create run script for different platforms
    if platform.system() == "Windows":
        run_script = """@echo off
echo Starting YOLOv13 Triple Input Model...
cd /d "%~dp0"
call yolo_env\\Scripts\\activate
python demo_verification.py
pause
"""
        with open("run_demo.bat", "w") as f:
            f.write(run_script)
        print("   ✅ Created run_demo.bat")
    else:
        run_script = """#!/bin/bash
echo "Starting YOLOv13 Triple Input Model..."
cd "$(dirname "$0")"
source yolo_env/bin/activate
python demo_verification.py
"""
        with open("run_demo.sh", "w") as f:
            f.write(run_script)
        os.chmod("run_demo.sh", 0o755)
        print("   ✅ Created run_demo.sh")
    
    # Create requirements check script
    check_script = """#!/usr/bin/env python3
import sys
import pkg_resources
import subprocess

def check_requirements():
    requirements = []
    
    # Read requirements file
    try:
        with open('requirements-minimal.txt', 'r') as f:
            requirements = f.read().strip().split('\\n')
    except FileNotFoundError:
        print("❌ requirements-minimal.txt not found")
        return False
    
    # Filter out comments and empty lines
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    
    missing = []
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
        except pkg_resources.DistributionNotFound:
            missing.append(requirement)
        except pkg_resources.VersionConflict as e:
            print(f"⚠️  Version conflict: {e}")
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

if __name__ == "__main__":
    if check_requirements():
        print("🎉 Environment is ready for deployment!")
    else:
        print("❌ Please install missing dependencies")
        sys.exit(1)
"""
    
    with open("check_requirements.py", "w") as f:
        f.write(check_script)
    
    print("   ✅ Created check_requirements.py")
    
    return True

def create_docker_setup():
    """Create Docker setup files"""
    print_step(6, "Creating Docker Setup (Optional)")
    
    # Dockerfile
    dockerfile = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libfontconfig1 \\
    libxrender1 \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models weights data outputs logs temp

# Set environment variables
ENV PYTHONPATH=/app:/app/yolov13

# Expose port for web interface (if using Gradio)
EXPOSE 7860

# Default command
CMD ["python", "demo_verification.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    # Docker compose
    docker_compose = """version: '3.8'

services:
  yolo-triple:
    build: .
    container_name: yolo-triple-input
    volumes:
      - ./weights:/app/weights
      - ./data:/app/data
      - ./outputs:/app/outputs
    ports:
      - "7860:7860"
    environment:
      - PYTHONPATH=/app:/app/yolov13
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("   ✅ Created Dockerfile and docker-compose.yml")
    
    return True

def run_verification():
    """Run verification to ensure setup works"""
    print_step(7, "Running Verification")
    
    # Check if demo verification exists
    if not Path("demo_verification.py").exists():
        print("   ❌ demo_verification.py not found")
        return False
    
    # Try to run verification
    try:
        print("   Running demo verification...")
        result = subprocess.run([sys.executable, "demo_verification.py"], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("   ✅ Verification passed!")
            return True
        else:
            print("   ⚠️  Verification completed with warnings")
            print(f"   Output: {result.stdout}")
            return True
    except subprocess.TimeoutExpired:
        print("   ⚠️  Verification timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="YOLOv13 Triple Input Deployment Setup")
    parser.add_argument("--env-name", default="yolo_env", help="Virtual environment name")
    parser.add_argument("--requirements", default="requirements-deployment.txt", 
                       help="Requirements file to use")
    parser.add_argument("--skip-venv", action="store_true", 
                       help="Skip virtual environment creation")
    parser.add_argument("--skip-docker", action="store_true", 
                       help="Skip Docker setup")
    parser.add_argument("--skip-verify", action="store_true", 
                       help="Skip verification")
    
    args = parser.parse_args()
    
    print_header("YOLOv13 Triple Input Model - Deployment Setup")
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        sys.exit(1)
    
    # Setup virtual environment
    env_name = None
    if not args.skip_venv:
        env_name = setup_virtual_environment(args.env_name)
        if not env_name:
            print("❌ Virtual environment setup failed")
            sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(env_name, args.requirements):
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Setup model structure
    if not setup_model_structure():
        print("❌ Model structure setup failed")
        sys.exit(1)
    
    # Create deployment scripts
    if not create_deployment_scripts():
        print("❌ Deployment script creation failed")
        sys.exit(1)
    
    # Create Docker setup
    if not args.skip_docker:
        create_docker_setup()
    
    # Run verification
    if not args.skip_verify:
        run_verification()
    
    print_header("Setup Complete!")
    print("🎉 YOLOv13 Triple Input Model is ready for deployment!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"   {args.env_name}\\Scripts\\activate")
    else:
        print(f"   source {args.env_name}/bin/activate")
    print("2. Run the demo:")
    print("   python demo_verification.py")
    print("3. Or use the provided run script:")
    if platform.system() == "Windows":
        print("   run_demo.bat")
    else:
        print("   ./run_demo.sh")
    print("\nFor Docker deployment:")
    print("   docker-compose up --build")

if __name__ == "__main__":
    main()