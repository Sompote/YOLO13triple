#!/usr/bin/env python3
"""
Enhanced Triple YOLO Model with pretrained weights support and multiple variants
Supports YOLOv13n, YOLOv13s, YOLOv13m, YOLOv13l, YOLOv13x variants
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from typing import Union, Optional, Dict, Any
import urllib.request
from collections import OrderedDict

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent.parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

from ultralytics.nn.modules.conv import TripleInputConv, Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules import Concat
from ultralytics.utils.torch_utils import initialize_weights


# Model scaling configurations
MODEL_VARIANTS = {
    'n': {'depth_multiple': 0.33, 'width_multiple': 0.25, 'max_channels': 1024},  # Nano
    's': {'depth_multiple': 0.33, 'width_multiple': 0.50, 'max_channels': 1024},  # Small  
    'm': {'depth_multiple': 0.67, 'width_multiple': 0.75, 'max_channels': 768},   # Medium
    'l': {'depth_multiple': 1.00, 'width_multiple': 1.00, 'max_channels': 512},   # Large
    'x': {'depth_multiple': 1.33, 'width_multiple': 1.25, 'max_channels': 512},   # Extra Large
}

# Pretrained weights URLs (these would need to be actual URLs to YOLOv13 weights)
PRETRAINED_URLS = {
    'yolov13n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov13n.pt',
    'yolov13s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov13s.pt',
    'yolov13m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov13m.pt',
    'yolov13l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov13l.pt',
    'yolov13x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov13x.pt',
}


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class EnhancedTripleYOLOModel(nn.Module):
    """Enhanced Triple YOLO model with variant support and pretrained weights."""
    
    def __init__(self, variant='n', nc=80, pretrained=None, freeze_backbone=False):
        """
        Initialize Enhanced Triple YOLO model.
        
        Args:
            variant (str): Model variant ('n', 's', 'm', 'l', 'x')
            nc (int): Number of classes
            pretrained (str or Path): Path to pretrained weights or variant name
            freeze_backbone (bool): Whether to freeze backbone for transfer learning
        """
        super().__init__()
        
        self.variant = variant
        self.nc = nc
        self.freeze_backbone = freeze_backbone
        
        # Get scaling parameters
        if variant in MODEL_VARIANTS:
            self.depth_multiple = MODEL_VARIANTS[variant]['depth_multiple']
            self.width_multiple = MODEL_VARIANTS[variant]['width_multiple']
            self.max_channels = MODEL_VARIANTS[variant]['max_channels']
        else:
            raise ValueError(f"Unsupported variant: {variant}. Choose from {list(MODEL_VARIANTS.keys())}")
        
        # Build model architecture
        self._build_model()
        
        # Initialize weights
        initialize_weights(self)
        
        # Load pretrained weights if specified
        if pretrained:
            self.load_pretrained(pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _make_layers(self, channels_in, channels_out, num_blocks=1):
        """Create scaled layers based on variant."""
        channels_out = min(self.max_channels, int(channels_out * self.width_multiple))
        num_blocks = max(1, int(num_blocks * self.depth_multiple))
        return channels_out, num_blocks
    
    def _build_model(self):
        """Build model architecture based on variant."""
        
        # Backbone with triple input and scaling
        ch64, _ = self._make_layers(3, 64)
        ch128, _ = self._make_layers(ch64, 128)
        ch256, blocks1 = self._make_layers(ch128, 256, 3)
        ch512, blocks2 = self._make_layers(ch256, 512, 6)
        ch1024, blocks3 = self._make_layers(ch512, 1024, 3)
        
        # Triple input layer (always uses base channels for compatibility)
        self.conv0 = TripleInputConv(3, 64, 3, 2)
        
        # Scaled backbone layers
        self.conv1 = Conv(64, ch128, 3, 2)
        self.c2f1 = C2f(ch128, ch128, blocks1)
        
        self.conv2 = Conv(ch128, ch256, 3, 2)
        self.c2f2 = C2f(ch256, ch256, blocks2)
        
        self.conv3 = Conv(ch256, ch512, 3, 2)
        self.c2f3 = C2f(ch512, ch512, blocks2)
        
        self.conv4 = Conv(ch512, ch1024, 3, 2)
        self.c2f4 = C2f(ch1024, ch1024, blocks3)
        
        self.sppf = SPPF(ch1024, ch1024, 5)
        
        # Scaled head layers
        head_ch512, head_blocks = self._make_layers(512, 512, 3)
        head_ch256, head_blocks = self._make_layers(256, 256, 3)
        
        # Head with upsampling path
        self.upsample1 = nn.Upsample(None, 2, "nearest")
        self.concat1 = Concat(1)
        self.c2f_head1 = C2f(ch1024 + ch512, head_ch512, head_blocks)
        
        self.upsample2 = nn.Upsample(None, 2, "nearest")
        self.concat2 = Concat(1)
        self.c2f_head2 = C2f(head_ch512 + ch256, head_ch256, head_blocks)
        
        # Downsampling path
        self.conv_head1 = Conv(head_ch256, head_ch256, 3, 2)
        self.concat3 = Concat(1)
        self.c2f_head3 = C2f(head_ch256 + head_ch512, head_ch512, head_blocks)
        
        self.conv_head2 = Conv(head_ch512, head_ch512, 3, 2)
        self.concat4 = Concat(1)
        self.c2f_head4 = C2f(head_ch512 + ch1024, ch1024, head_blocks)
        
        # Detection heads for P3, P4, P5
        self.detect = Detect(self.nc, [head_ch256, head_ch512, ch1024])
        
        # Store channel dimensions for weight loading
        self.backbone_channels = {
            'conv1': ch128,
            'c2f1': ch128,
            'conv2': ch256,
            'c2f2': ch256,
            'conv3': ch512,
            'c2f3': ch512,
            'conv4': ch1024,
            'c2f4': ch1024,
            'sppf': ch1024,
        }
    
    def forward(self, x):
        """Forward pass with triple input support."""
        # Backbone
        x0 = self.conv0(x)  # Triple input processing
        x1 = self.conv1(x0)
        x1 = self.c2f1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.c2f2(x2)  # P3
        
        x3 = self.conv3(x2)
        x3 = self.c2f3(x3)  # P4
        
        x4 = self.conv4(x3)
        x4 = self.c2f4(x4)
        x4 = self.sppf(x4)  # P5
        
        # Head
        p5 = x4
        p4_up = self.upsample1(p5)
        p4 = self.concat1([p4_up, x3])
        p4 = self.c2f_head1(p4)
        
        p3_up = self.upsample2(p4)
        p3 = self.concat2([p3_up, x2])
        p3 = self.c2f_head2(p3)  # Small objects
        
        # Downsample path
        p4_down = self.conv_head1(p3)
        p4 = self.concat3([p4_down, p4])
        p4 = self.c2f_head3(p4)  # Medium objects
        
        p5_down = self.conv_head2(p4)
        p5 = self.concat4([p5_down, p5])
        p5 = self.c2f_head4(p5)  # Large objects
        
        # Detection
        return self.detect([p3, p4, p5])
    
    def load_pretrained(self, weights_path_or_name):
        """
        Load pretrained weights.
        
        Args:
            weights_path_or_name (str): Path to weights file or variant name for download
        """
        print(f"Loading pretrained weights: {weights_path_or_name}")
        
        # Handle different input types
        if weights_path_or_name in PRETRAINED_URLS:
            # Download weights if it's a known variant
            weights_path = self._download_weights(weights_path_or_name)
        else:
            # Use provided path
            weights_path = Path(weights_path_or_name)
        
        if not weights_path.exists():
            print(f"Warning: Pretrained weights not found at {weights_path}")
            print("Continuing with random initialization...")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Extract model state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Convert to float (in case of float16)
            if hasattr(state_dict, 'float'):
                state_dict = state_dict.float().state_dict()
            elif isinstance(state_dict, dict):
                for key in state_dict:
                    if hasattr(state_dict[key], 'float'):
                        state_dict[key] = state_dict[key].float()
            
            # Load compatible weights
            self._load_compatible_state_dict(state_dict)
            
            print(f"✅ Successfully loaded pretrained weights from {weights_path}")
            
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")
            print("Continuing with random initialization...")
    
    def _load_compatible_state_dict(self, state_dict):
        """Load state dict with compatibility handling for triple input."""
        
        model_dict = self.state_dict()
        
        # Skip the first layer (conv0) since it's modified for triple input
        pretrained_dict = {}
        
        for name, param in state_dict.items():
            # Skip triple input layer (we'll handle it separately)
            if 'conv0' in name or '0.conv' in name or '0.bn' in name:
                continue
            
            # Handle layer name mapping if needed
            mapped_name = self._map_layer_name(name)
            
            if mapped_name in model_dict:
                if model_dict[mapped_name].shape == param.shape:
                    pretrained_dict[mapped_name] = param
                else:
                    print(f"Shape mismatch for {mapped_name}: "
                          f"model {model_dict[mapped_name].shape} vs pretrained {param.shape}")
            else:
                print(f"Layer {mapped_name} not found in model")
        
        # Load the compatible weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
    
    def _map_layer_name(self, name):
        """Map pretrained layer names to current model names."""
        # Handle common name mappings between different YOLO versions
        name_mapping = {
            'model.0': 'conv0',  # Skip this - handled separately
            'model.1': 'conv1',
            'model.2': 'c2f1',
            'model.3': 'conv2',
            'model.4': 'c2f2',
            'model.5': 'conv3',
            'model.6': 'c2f3',
            'model.7': 'conv4',
            'model.8': 'c2f4',
            'model.9': 'sppf',
            # Add more mappings as needed
        }
        
        for old_name, new_name in name_mapping.items():
            if name.startswith(old_name):
                return name.replace(old_name, new_name)
        
        return name
    
    def _download_weights(self, variant_name):
        """Download pretrained weights for specified variant."""
        if variant_name not in PRETRAINED_URLS:
            raise ValueError(f"No pretrained weights available for {variant_name}")
        
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / f"{variant_name}.pt"
        
        if weights_path.exists():
            print(f"Using cached weights: {weights_path}")
            return weights_path
        
        print(f"Downloading {variant_name} weights...")
        try:
            urllib.request.urlretrieve(PRETRAINED_URLS[variant_name], weights_path)
            print(f"Downloaded weights to: {weights_path}")
            return weights_path
        except Exception as e:
            print(f"Failed to download weights: {e}")
            return weights_path  # Return path anyway, error will be handled in load_pretrained
    
    def _freeze_backbone(self):
        """Freeze backbone layers for transfer learning."""
        print("Freezing backbone layers...")
        frozen_layers = []
        
        backbone_modules = [
            'conv1', 'c2f1', 'conv2', 'c2f2', 
            'conv3', 'c2f3', 'conv4', 'c2f4', 'sppf'
        ]
        
        for name, param in self.named_parameters():
            if any(module in name for module in backbone_modules):
                param.requires_grad = False
                frozen_layers.append(name)
        
        print(f"Frozen {len(frozen_layers)} backbone parameters")
        print("Head layers remain trainable for task-specific learning")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone layers."""
        print("Unfreezing backbone layers...")
        unfrozen_layers = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_layers.append(name)
        
        print(f"Unfrozen {len(unfrozen_layers)} parameters")
    
    def get_model_info(self):
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'variant': self.variant,
            'depth_multiple': self.depth_multiple,
            'width_multiple': self.width_multiple,
            'num_classes': self.nc,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'backbone_frozen': self.freeze_backbone,
            'backbone_channels': self.backbone_channels,
        }
        
        return info
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        
        print(f"\n📊 Model Information:")
        print(f"   Variant: YOLOv13{info['variant']}")
        print(f"   Depth Multiple: {info['depth_multiple']}")
        print(f"   Width Multiple: {info['width_multiple']}")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Total Parameters: {info['total_parameters']:,}")
        print(f"   Trainable Parameters: {info['trainable_parameters']:,}")
        if info['frozen_parameters'] > 0:
            print(f"   Frozen Parameters: {info['frozen_parameters']:,}")
        print()


def create_triple_yolo_model(variant='n', nc=80, pretrained=None, freeze_backbone=False):
    """
    Create a triple input YOLO model.
    
    Args:
        variant (str): Model variant ('n', 's', 'm', 'l', 'x')
        nc (int): Number of classes
        pretrained (str): Pretrained weights path or variant name
        freeze_backbone (bool): Whether to freeze backbone
    
    Returns:
        EnhancedTripleYOLOModel: The created model
    """
    return EnhancedTripleYOLOModel(
        variant=variant,
        nc=nc,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


# Convenience functions for different variants
def triple_yolo13n(nc=80, pretrained=None, freeze_backbone=False):
    """Create YOLOv13n triple input model."""
    return create_triple_yolo_model('n', nc, pretrained, freeze_backbone)

def triple_yolo13s(nc=80, pretrained=None, freeze_backbone=False):
    """Create YOLOv13s triple input model."""
    return create_triple_yolo_model('s', nc, pretrained, freeze_backbone)

def triple_yolo13m(nc=80, pretrained=None, freeze_backbone=False):
    """Create YOLOv13m triple input model."""
    return create_triple_yolo_model('m', nc, pretrained, freeze_backbone)

def triple_yolo13l(nc=80, pretrained=None, freeze_backbone=False):
    """Create YOLOv13l triple input model."""
    return create_triple_yolo_model('l', nc, pretrained, freeze_backbone)

def triple_yolo13x(nc=80, pretrained=None, freeze_backbone=False):
    """Create YOLOv13x triple input model."""
    return create_triple_yolo_model('x', nc, pretrained, freeze_backbone)


if __name__ == "__main__":
    # Example usage
    print("Testing Enhanced Triple YOLO Models:")
    
    variants = ['n', 's', 'm']
    
    for variant in variants:
        print(f"\n🧪 Testing YOLOv13{variant} Triple Input:")
        model = create_triple_yolo_model(variant=variant, nc=80)
        model.print_model_info()
        
        # Test forward pass
        dummy_images = [torch.randn(1, 3, 640, 640) for _ in range(3)]
        with torch.no_grad():
            output = model(dummy_images)
        
        print(f"   Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
        print(f"   ✅ YOLOv13{variant} triple input working!")