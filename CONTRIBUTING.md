# Contributing to YOLOv13 Triple Input

Thank you for your interest in contributing to the YOLOv13 Triple Input project! 🎉

We welcome contributions from the community and are grateful for any help you can provide, whether it's:
- 🐛 **Bug reports**
- 💡 **Feature requests** 
- 📝 **Documentation improvements**
- 🧪 **Code contributions**
- 🧹 **Code reviews**

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them learn
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together towards common goals

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:
- Python 3.8+ installed
- Git installed and configured
- Basic understanding of PyTorch and computer vision
- Familiarity with YOLO architectures (helpful but not required)

### Quick Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/yolo-triple-input.git
cd yolo-triple-input

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/yolo-triple-input.git

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run tests to verify setup
python test_triple_implementation.py
```

## 🛠️ Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Additional dev tools

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Project Structure

```
yolo_3dual_input/
├── yolov13/ultralytics/          # Core YOLO framework
│   ├── nn/modules/conv.py        # TripleInputConv module
│   ├── data/triple_dataset.py    # Dataset handling
│   └── cfg/models/v13/           # Model configurations
├── triple_inference.py           # Main inference script
├── train_direct_triple.py        # Training pipeline
├── test_*.py                     # Test files
├── docs/                         # Documentation
└── examples/                     # Usage examples
```

## 🤝 How to Contribute

### 1. Choose Your Contribution Type

#### 🐛 Bug Reports
Found a bug? Please help us fix it!

**Before submitting:**
- Check if the issue already exists
- Make sure you're using the latest version
- Try to reproduce the bug consistently

**What to include:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior  
- Environment information (OS, Python version, etc.)
- Error messages and stack traces
- Sample code or data (if applicable)

#### 💡 Feature Requests
Have an idea for improvement?

**Before submitting:**
- Check if the feature was already requested
- Consider if it fits the project's scope
- Think about implementation challenges

**What to include:**
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Alternatives considered

#### 🧪 Code Contributions
Ready to write some code?

**Good first issues:**
- Documentation improvements
- Adding type hints
- Writing tests
- Performance optimizations
- Bug fixes

**Larger contributions:**
- New fusion algorithms
- Export functionality (ONNX, TensorRT)
- Mobile optimizations
- Web interface development

### 2. Fork and Branch

```bash
# Keep your fork updated
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. Make Your Changes

Follow our coding standards and write tests for new functionality.

### 4. Test Your Changes

```bash
# Run full test suite
python test_triple_implementation.py

# Test specific functionality
python triple_inference.py --primary sample_data/primary/image_1.jpg --detail1 sample_data/detail1/image_1.jpg --detail2 sample_data/detail2/image_1.jpg

# Run training test
python train_direct_triple.py --data-dir training_data_demo --epochs 1

# Check code style
black . --check
flake8 .
```

## 📝 Pull Request Process

### Before Submitting

1. **Update Documentation**: Update README, docstrings, and comments
2. **Add Tests**: Include tests for new functionality
3. **Check Compatibility**: Ensure backward compatibility
4. **Update Changelog**: Add entry to CHANGELOG.md
5. **Rebase**: Rebase your branch on latest main

### PR Submission

1. **Title**: Use clear, descriptive title
   - `feat: add ONNX export functionality`
   - `fix: resolve memory leak in triple input processing`
   - `docs: update installation instructions`

2. **Description**: Include:
   - What changes were made
   - Why they were made
   - How to test the changes
   - Any breaking changes
   - Related issues

3. **Checklist**: Use our PR template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI/CD will run tests
2. **Code Review**: Maintainers will review code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, PR will be merged

## 📏 Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) with these additions:

```python
# Use type hints
def process_images(images: List[torch.Tensor]) -> torch.Tensor:
    """Process multiple images and return fused features."""
    pass

# Document functions clearly
def triple_forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Forward pass for triple input.
    
    Args:
        x: Either single tensor or list of 3 tensors [primary, detail1, detail2]
        
    Returns:
        Processed feature tensor with attention-weighted fusion
        
    Raises:
        ValueError: If input format is invalid
    """
    pass

# Use meaningful variable names
primary_features = self.conv1(primary_image)
detail1_features = self.conv2(detail1_image)
```

### Code Organization

```python
# Import order
import os  # Standard library
import sys
from pathlib import Path

import torch  # Third party
import numpy as np
import cv2

from ultralytics.nn.modules import Conv  # Local imports
```

### Error Handling

```python
# Provide helpful error messages
def load_triple_images(paths):
    if len(paths) != 3:
        raise ValueError(
            f"Expected exactly 3 image paths, got {len(paths)}. "
            f"Please provide paths for [primary, detail1, detail2] images."
        )
    
    # Graceful fallbacks
    for i, path in enumerate(paths):
        if not Path(path).exists():
            if i == 0:  # Primary image is required
                raise FileNotFoundError(f"Primary image not found: {path}")
            else:  # Detail images can fallback
                logger.warning(f"Detail image {i} not found, using primary image")
                paths[i] = paths[0]
```

## 🧪 Testing Guidelines

### Test Structure

```python
# test_triple_input.py
import pytest
import torch
from ultralytics.nn.modules.conv import TripleInputConv

class TestTripleInputConv:
    def test_triple_input_forward(self):
        """Test forward pass with triple inputs."""
        conv = TripleInputConv(3, 64, 3, 2)
        
        # Create test inputs
        batch_size = 2
        inputs = [
            torch.randn(batch_size, 3, 640, 640),  # Primary
            torch.randn(batch_size, 3, 640, 640),  # Detail1  
            torch.randn(batch_size, 3, 640, 640),  # Detail2
        ]
        
        # Test forward pass
        output = conv(inputs)
        
        # Assertions
        assert output.shape == (batch_size, 64, 320, 320)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
    
    def test_single_input_fallback(self):
        """Test fallback to single input."""
        conv = TripleInputConv(3, 64, 3, 2)
        
        # Single input
        single_input = torch.randn(2, 3, 640, 640)
        output = conv(single_input)
        
        assert output.shape[1] == 64  # Output channels
```

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark speed and memory
4. **Regression Tests**: Ensure bug fixes stay fixed

### Running Tests

```bash
# Run all tests
python test_triple_implementation.py

# Run specific test
python -m pytest test_triple_input.py::TestTripleInputConv::test_triple_input_forward

# Run with coverage
python -m pytest --cov=. test_*.py
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def attention_fusion(self, features: torch.Tensor) -> torch.Tensor:
    """
    Apply attention-based fusion to concatenated features.
    
    This method computes channel-wise attention weights and applies them
    to the input features for adaptive feature selection.
    
    Args:
        features: Concatenated feature tensor of shape (B, C, H, W)
        
    Returns:
        Attention-weighted feature tensor of same shape as input
        
    Example:
        >>> conv = TripleInputConv(3, 64)
        >>> features = torch.randn(1, 64, 320, 320)
        >>> weighted = conv.attention_fusion(features)
        >>> assert weighted.shape == features.shape
    """
```

### Documentation Updates

When adding features, update:
- **README.md**: User-facing documentation
- **API docs**: Function/class documentation  
- **Examples**: Usage examples
- **Changelog**: Version history

## 🏷️ Issues and Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation needs
- `good first issue`: Good for newcomers
- `help wanted`: Community help needed
- `performance`: Performance related
- `training`: Training pipeline related
- `inference`: Inference related

## 🌟 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Added to CONTRIBUTORS.md
- Recognized in project documentation

## 💬 Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

### Getting Help

1. **Check Documentation**: README, docs folder, code comments
2. **Search Issues**: Look for similar problems
3. **Ask Questions**: Use GitHub Discussions
4. **Join Community**: Participate in discussions

### Mentoring

New contributors can:
- Start with `good first issue` labels
- Ask for guidance in issues
- Request code review feedback
- Pair program with maintainers

## 🎯 Contribution Areas

### High Priority
- [ ] **Performance Optimization**: CUDA kernels, mixed precision
- [ ] **Export Support**: ONNX, TensorRT conversion
- [ ] **Documentation**: API docs, tutorials
- [ ] **Testing**: Increase test coverage
- [ ] **Examples**: Real-world use cases

### Medium Priority  
- [ ] **Mobile Support**: Quantization, optimization
- [ ] **Web Interface**: Gradio/Streamlit demo
- [ ] **Data Augmentation**: Triple-aware transforms
- [ ] **Visualization**: Better result analysis
- [ ] **Benchmarks**: Performance comparisons

### Future Features
- [ ] **Multi-scale Training**: Different input resolutions
- [ ] **Temporal Fusion**: Video processing
- [ ] **Cross-modal**: RGB + thermal/depth
- [ ] **AutoML**: Architecture search

## 📞 Contact

- **Maintainer**: [Your Name] - your.email@example.com
- **Project**: https://github.com/yourusername/yolo-triple-input
- **Issues**: https://github.com/yourusername/yolo-triple-input/issues

---

Thank you for contributing to YOLOv13 Triple Input! 🙏

*Every contribution, no matter how small, makes a difference.*