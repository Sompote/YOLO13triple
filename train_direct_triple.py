#!/usr/bin/env python3
"""
Direct training script for triple input YOLO model
Uses custom training loop for better control over triple inputs
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
from torch.utils.data import Dataset, DataLoader
import argparse

# Add the yolov13 directory to Python path
yolov13_path = Path(__file__).parent / "yolov13"
sys.path.insert(0, str(yolov13_path))

from triple_inference import TripleYOLOModel, load_and_preprocess_images


class TripleDataset(Dataset):
    """Dataset for triple image training."""
    
    def __init__(self, data_dir, split='train', imgsz=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.imgsz = imgsz
        
        # Get primary image files
        primary_dir = self.data_dir / 'images' / 'primary' / split
        self.primary_files = list(primary_dir.glob('*.jpg')) + list(primary_dir.glob('*.png'))
        
        # Get corresponding detail files
        detail1_dir = self.data_dir / 'images' / 'detail1' / split
        detail2_dir = self.data_dir / 'images' / 'detail2' / split
        
        self.detail1_files = []
        self.detail2_files = []
        self.label_files = []
        
        labels_dir = self.data_dir / 'labels' / split
        
        for primary_file in self.primary_files:
            # Find corresponding files
            filename = primary_file.name
            stem = primary_file.stem
            
            detail1_file = detail1_dir / filename
            detail2_file = detail2_dir / filename
            label_file = labels_dir / f"{stem}.txt"
            
            if detail1_file.exists() and detail2_file.exists() and label_file.exists():
                self.detail1_files.append(detail1_file)
                self.detail2_files.append(detail2_file)
                self.label_files.append(label_file)
            else:
                print(f"Missing files for {filename}, skipping...")
        
        # Update primary files to only include those with complete sets
        self.primary_files = self.primary_files[:len(self.label_files)]
        
        print(f"Loaded {len(self.primary_files)} samples for {split}")
    
    def __len__(self):
        return len(self.primary_files)
    
    def load_labels(self, label_file):
        """Load YOLO format labels."""
        labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            labels.append([class_id, x, y, w, h])
        
        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
    
    def __getitem__(self, idx):
        # Load triple images
        primary_path = str(self.primary_files[idx])
        detail1_path = str(self.detail1_files[idx])
        detail2_path = str(self.detail2_files[idx])
        
        try:
            images = load_and_preprocess_images(primary_path, detail1_path, detail2_path, self.imgsz)
        except Exception as e:
            print(f"Error loading images for index {idx}: {e}")
            # Return dummy data
            dummy_img = torch.zeros(3, self.imgsz, self.imgsz)
            images = [dummy_img, dummy_img, dummy_img]
        
        # Load labels
        labels = self.load_labels(self.label_files[idx])
        
        return {
            'images': images,
            'labels': labels,
            'img_path': primary_path
        }


def collate_fn(batch):
    """Custom collate function for triple inputs."""
    images = []
    labels = []
    paths = []
    
    for item in batch:
        images.append(item['images'])
        labels.append(item['labels'])
        paths.append(item['img_path'])
    
    return {
        'images': images,
        'labels': labels,
        'paths': paths
    }


def compute_loss(predictions, targets, device):
    """Simple loss computation for demonstration."""
    # This is a simplified loss - in practice you'd use proper YOLO loss
    if isinstance(predictions, tuple):
        if isinstance(predictions[0], torch.Tensor):
            pred_tensor = predictions[0]  # Use inference output
        else:
            # If it's a list of tensors, take the first one
            pred_tensor = predictions[1][0] if isinstance(predictions[1], list) else predictions[0]
    else:
        pred_tensor = predictions
    
    # For demonstration, just compute a dummy loss
    # In real implementation, you'd compute proper objectness, classification, and box regression losses
    if isinstance(pred_tensor, torch.Tensor):
        dummy_loss = torch.mean(torch.abs(pred_tensor)) * 0.001  # Very small loss to avoid numerical issues
    else:
        # Fallback
        dummy_loss = torch.tensor(0.001, device=device, requires_grad=True)
    
    return dummy_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Process batch
        batch_images = batch['images']
        batch_labels = batch['labels']
        
        # Move to device and prepare inputs
        processed_inputs = []
        for sample_images in batch_images:
            # Each sample has 3 images
            sample_on_device = [img.to(device) for img in sample_images]
            processed_inputs.append(sample_on_device)
        
        # Forward pass for each sample in batch
        total_batch_loss = 0.0
        for sample_inputs, sample_labels in zip(processed_inputs, batch_labels):
            # Add batch dimension
            batched_inputs = [img.unsqueeze(0) for img in sample_inputs]
            
            # Forward pass
            predictions = model(batched_inputs)
            
            # Compute loss
            sample_labels = sample_labels.to(device)
            loss = compute_loss(predictions, sample_labels, device)
            total_batch_loss += loss
        
        # Average loss over batch
        avg_batch_loss = total_batch_loss / len(batch_images)
        
        # Backward pass
        avg_batch_loss.backward()
        optimizer.step()
        
        total_loss += avg_batch_loss.item()
        num_batches += 1
        
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {avg_batch_loss.item():.6f}")
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_images = batch['images']
            batch_labels = batch['labels']
            
            # Process batch (similar to training)
            processed_inputs = []
            for sample_images in batch_images:
                sample_on_device = [img.to(device) for img in sample_images]
                processed_inputs.append(sample_on_device)
            
            total_batch_loss = 0.0
            for sample_inputs, sample_labels in zip(processed_inputs, batch_labels):
                batched_inputs = [img.unsqueeze(0) for img in sample_inputs]
                predictions = model(batched_inputs)
                sample_labels = sample_labels.to(device)
                loss = compute_loss(predictions, sample_labels, device)
                total_batch_loss += loss
            
            avg_batch_loss = total_batch_loss / len(batch_images)
            total_loss += avg_batch_loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Direct Triple Input YOLO Training")
    parser.add_argument("--data-dir", type=str, default="training_data_demo", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--save-dir", type=str, default="runs/train_direct", help="Save directory")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TripleDataset(args.data_dir, 'train', args.imgsz)
    val_dataset = TripleDataset(args.data_dir, 'val', args.imgsz)
    
    if len(train_dataset) == 0:
        print("No training data found!")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    ) if len(val_dataset) > 0 else None
    
    # Create model
    print("Creating model...")
    model = TripleYOLOModel(nc=80)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)
        print(f"Training Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_dir / 'best.pt')
                print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss if val_loader else 0.0,
        }, save_dir / 'latest.pt')
    
    print(f"\n✅ Training completed!")
    print(f"Models saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Test inference with trained model
    print("\n🧪 Testing inference with trained model...")
    model.eval()
    
    # Get a sample from training data
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        images = sample['images']
        
        with torch.no_grad():
            batched_inputs = [img.unsqueeze(0).to(device) for img in images]
            predictions = model(batched_inputs)
            
            if isinstance(predictions, tuple):
                pred_shape = predictions[0].shape
            else:
                pred_shape = predictions.shape
            
            print(f"✅ Inference test passed! Output shape: {pred_shape}")


if __name__ == "__main__":
    main()