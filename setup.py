#!/usr/bin/env python3
"""
Setup script for YOLOv13 Triple Input package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'opencv-python>=4.0.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'pyyaml>=5.4.0',
    ]

setup(
    name="yolov13-triple-input",
    version="1.0.0",
    author="YOLOv13 Triple Input Contributors",
    author_email="your.email@example.com",
    description="YOLOv13 implementation for processing 3 images simultaneously with attention-based fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yolo-triple-input",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/yolo-triple-input/issues",
        "Source": "https://github.com/yourusername/yolo-triple-input",
        "Documentation": "https://github.com/yourusername/yolo-triple-input/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.15.0",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "full": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.15.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolo-triple-train=train_direct_triple:main",
            "yolo-triple-infer=triple_inference:main",
            "yolo-triple-test=test_triple_implementation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
        "yolov13": ["ultralytics/cfg/**/*.yaml"],
    },
    keywords=[
        "yolo",
        "object-detection", 
        "computer-vision",
        "deep-learning",
        "pytorch",
        "triple-input",
        "multi-image",
        "attention-fusion",
        "machine-learning",
        "ai",
    ],
    license="AGPL-3.0",
    zip_safe=False,
)