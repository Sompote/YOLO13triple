#!/usr/bin/env python3
"""
Enhanced Training Script for Triple Input YOLO with Pretrained Weights Support
Supports transfer learning, model variants, and advanced training configurations
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import yaml
import argparse
from torch.utils.data import Dataset, DataLoader
import time
from typing import Optional, Dict, Any

# Add project paths
project_root = Path(__file__).parent
yolov13_path = project_root / "yolov13"
sys.path.insert(0, str(yolov13_path))
sys.path.insert(0, str(project_root))

from models.triple_yolo_variants import create_triple_yolo_model, MODEL_VARIANTS
from train_direct_triple import TripleDataset, collate_fn


class EnhancedTripleTrainer:
    """Enhanced trainer with pretrained weights and transfer learning support."""
    
    def __init__(self, config):
        """
        Initialize enhanced trainer.
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'auto'))
        if str(self.device) == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Enhanced Triple YOLO Training")
        print(f"   Variant: YOLOv13{config['variant']}")
        print(f"   Device: {self.device}")
        print(f"   Pretrained: {config.get('pretrained', 'None')}")
        print(f"   Transfer Learning: {config.get('freeze_backbone', False)}")
        
        # Create model
        self.model = self._create_model()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # Resume from checkpoint if specified
        if config.get('resume'):
            self._resume_training()
    
    def _create_model(self):
        """Create model with pretrained weights."""
        config = self.config
        
        model = create_triple_yolo_model(
            variant=config['variant'],
            nc=config['nc'],
            pretrained=config.get('pretrained'),
            freeze_backbone=config.get('freeze_backbone', False)
        )
        
        model.to(self.device)
        model.print_model_info()
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different parts."""
        config = self.config
        
        # Separate parameters for different learning rates
        backbone_params = []
        head_params = []
        triple_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'conv0' in name:  # Triple input layer
                triple_params.append(param)
            elif any(x in name for x in ['conv1', 'conv2', 'conv3', 'conv4', 'c2f1', 'c2f2', 'c2f3', 'c2f4', 'sppf']):
                backbone_params.append(param)
            else:  # Head parameters
                head_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if triple_params:
            param_groups.append({
                'params': triple_params,
                'lr': config['lr'] * config.get('triple_lr_multiplier', 1.0),
                'name': 'triple_input'
            })
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': config['lr'] * config.get('backbone_lr_multiplier', 0.1 if config.get('pretrained') else 1.0),
                'name': 'backbone'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': config['lr'] * config.get('head_lr_multiplier', 1.0),
                'name': 'head'
            })
        
        # Create optimizer
        optimizer_name = config.get('optimizer', 'AdamW').lower()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 0.0005),
                betas=config.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=config['lr'],
                momentum=config.get('momentum', 0.937),
                weight_decay=config.get('weight_decay', 0.0005)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        print(f"📈 Optimizer: {optimizer_name}")
        for i, group in enumerate(param_groups):
            print(f"   Group {i+1} ({group['name']}): {len(group['params'])} params, lr={group['lr']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        config = self.config
        scheduler_name = config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config.get('min_lr', config['lr'] * 0.01)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif scheduler_name == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.get('gamma', 0.95)
            )
        else:
            scheduler = None
        
        print(f"📅 Scheduler: {scheduler_name if scheduler else 'None'}")
        return scheduler
    
    def _create_loss_function(self):
        """Create loss function (simplified for demo)."""
        # This is a simplified loss function for demonstration
        # In practice, you would implement proper YOLO loss with:
        # - Objectness loss
        # - Classification loss  
        # - Box regression loss
        # - Optional: segmentation loss
        
        def simple_yolo_loss(predictions, targets):
            """Simplified YOLO loss for demonstration."""
            if isinstance(predictions, tuple):
                if isinstance(predictions[0], torch.Tensor):
                    pred_tensor = predictions[0]
                else:
                    pred_tensor = predictions[1][0] if isinstance(predictions[1], list) else predictions[0]
            else:
                pred_tensor = predictions
            
            # Simple L1 loss on predictions (demo purposes)
            # In practice, compute proper objectness, classification, and bbox losses
            loss = torch.mean(torch.abs(pred_tensor)) * 0.001
            
            return loss
        
        return simple_yolo_loss
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        checkpoint_path = self.config['resume']
        print(f"📂 Resuming training from: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', 
                                             {'train_loss': [], 'val_loss': [], 'lr': []})
        
        # Load scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.6f}")
    
    def train_one_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\n📚 Training Epoch {epoch}/{self.config['epochs']}")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Process batch
            batch_images = batch['images']
            batch_labels = batch['labels']
            
            # Move to device and prepare inputs
            processed_inputs = []
            for sample_images in batch_images:
                sample_on_device = [img.to(self.device) for img in sample_images]
                processed_inputs.append(sample_on_device)
            
            # Forward pass for each sample in batch
            total_batch_loss = 0.0
            for sample_inputs, sample_labels in zip(processed_inputs, batch_labels):
                # Add batch dimension
                batched_inputs = [img.unsqueeze(0) for img in sample_inputs]
                
                # Forward pass
                predictions = self.model(batched_inputs)
                
                # Compute loss
                sample_labels = sample_labels.to(self.device)
                loss = self.loss_fn(predictions, sample_labels)
                total_batch_loss += loss
            
            # Average loss over batch
            avg_batch_loss = total_batch_loss / len(batch_images)
            
            # Backward pass
            avg_batch_loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += avg_batch_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={avg_batch_loss.item():.6f}, LR={current_lr:.6f}")
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"📊 Epoch {epoch} Training: Loss={avg_loss:.6f}, Time={epoch_time:.1f}s")
        
        return avg_loss
    
    def validate_one_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"🧪 Validating Epoch {epoch}")
        start_time = time.time()
        
        with torch.no_grad():
            for batch in val_loader:
                batch_images = batch['images']
                batch_labels = batch['labels']
                
                # Process batch
                processed_inputs = []
                for sample_images in batch_images:
                    sample_on_device = [img.to(self.device) for img in sample_images]
                    processed_inputs.append(sample_on_device)
                
                total_batch_loss = 0.0
                for sample_inputs, sample_labels in zip(processed_inputs, batch_labels):
                    batched_inputs = [img.unsqueeze(0) for img in sample_inputs]
                    predictions = self.model(batched_inputs)
                    sample_labels = sample_labels.to(self.device)
                    loss = self.loss_fn(predictions, sample_labels)
                    total_batch_loss += loss
                
                avg_batch_loss = total_batch_loss / len(batch_images)
                total_loss += avg_batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_time = time.time() - start_time
        
        print(f"📊 Epoch {epoch} Validation: Loss={avg_loss:.6f}, Time={val_time:.1f}s")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        latest_path = save_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
        
        # Save epoch checkpoint (every N epochs)
        if epoch % self.config.get('save_period', 10) == 0:
            epoch_path = save_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        print(f"\n🎯 Starting Enhanced Triple YOLO Training")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch Size: {self.config['batch_size']}")
        print(f"   Learning Rate: {self.config['lr']}")
        print(f"   Save Directory: {self.config['save_dir']}")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_loss = self.train_one_epoch(train_loader, epoch + 1)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                val_loss = self.validate_one_epoch(val_loader, epoch + 1)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update training history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['lr'].append(current_lr)
            
            # Check if best model
            is_best = val_loss < self.best_loss if val_loader else train_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss if val_loader else train_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            print(f"📈 Epoch {epoch + 1} Summary: "
                  f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"LR={current_lr:.6f}, Best={self.best_loss:.6f}")
        
        print(f"\n🎉 Training completed!")
        print(f"   Best loss: {self.best_loss:.6f}")
        print(f"   Model saved to: {self.config['save_dir']}")
        
        return self.training_history


def create_training_config(args):
    """Create training configuration from arguments."""
    config = {
        # Model configuration
        'variant': args.variant,
        'nc': args.nc,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone,
        
        # Training configuration
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        
        # Transfer learning
        'backbone_lr_multiplier': args.backbone_lr_multiplier,
        'head_lr_multiplier': args.head_lr_multiplier,
        'triple_lr_multiplier': args.triple_lr_multiplier,
        
        # Training settings
        'device': args.device,
        'grad_clip': args.grad_clip,
        'log_interval': args.log_interval,
        'save_period': args.save_period,
        
        # Data
        'data_dir': args.data_dir,
        'imgsz': args.imgsz,
        
        # Output
        'save_dir': args.save_dir,
        'resume': args.resume,
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Enhanced Triple Input YOLO Training")
    
    # Model configuration
    parser.add_argument("--variant", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="Model variant (default: n)")
    parser.add_argument("--nc", type=int, default=80, help="Number of classes (default: 80)")
    parser.add_argument("--pretrained", type=str, help="Pretrained weights path or variant name")
    parser.add_argument("--freeze-backbone", action="store_true", 
                       help="Freeze backbone for transfer learning")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (default: 0.0005)")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=['AdamW', 'SGD'],
                       help="Optimizer (default: AdamW)")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                       choices=['cosine', 'step', 'exponential', 'none'],
                       help="Scheduler (default: cosine)")
    
    # Transfer learning
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1,
                       help="Backbone learning rate multiplier for pretrained models (default: 0.1)")
    parser.add_argument("--head-lr-multiplier", type=float, default=1.0,
                       help="Head learning rate multiplier (default: 1.0)")
    parser.add_argument("--triple-lr-multiplier", type=float, default=1.0,
                       help="Triple input layer learning rate multiplier (default: 1.0)")
    
    # Training settings
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--grad-clip", type=float, help="Gradient clipping threshold")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval (default: 10)")
    parser.add_argument("--save-period", type=int, default=10, help="Save period (default: 10)")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    
    # Output
    parser.add_argument("--save-dir", type=str, default="runs/enhanced_train", 
                       help="Save directory (default: runs/enhanced_train)")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    
    # Utilities
    parser.add_argument("--validate-data", action="store_true", help="Validate dataset before training")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Validate dataset if requested
    if args.validate_data:
        from train_triple import validate_dataset
        if not validate_dataset(args.data_dir):
            print("❌ Dataset validation failed")
            return
        print("✅ Dataset validation passed")
    
    # Create training configuration
    config = create_training_config(args)
    
    # Create datasets
    print("📂 Creating datasets...")
    train_dataset = TripleDataset(args.data_dir, 'train', args.imgsz)
    val_dataset = TripleDataset(args.data_dir, 'val', args.imgsz)
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    print(f"📊 Dataset Info:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader) if val_loader else 0}")
    
    # Create trainer
    trainer = EnhancedTripleTrainer(config)
    
    # Start training
    try:
        training_history = trainer.train(train_loader, val_loader)
        print(f"✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print(f"   Latest checkpoint saved to: {config['save_dir']}/latest.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()