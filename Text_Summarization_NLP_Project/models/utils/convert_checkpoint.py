#!/usr/bin/env python3
"""
Convert training checkpoint to full model
"""

import torch
import shutil
from pathlib import Path

def convert_checkpoint(checkpoint_path: str, output_path: str):
    """
    Convert a training checkpoint to a full model.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Path to save converted model
    """
    checkpoint_dir = Path(checkpoint_path)
    output_dir = Path(output_path)
    
    # Check if checkpoint exists
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        return False
    
    # Look for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("pytorch_model.bin"))
    if not checkpoint_files:
        print(f"❌ No model file found in checkpoint")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Converting checkpoint: {checkpoint_path}")
    print(f"   Output: {output_path}")
    
    # Copy all files from checkpoint
    files_copied = 0
    for file in checkpoint_dir.rglob("*"):
        if file.is_file():
            # Create relative path
            relative_path = file.relative_to(checkpoint_dir)
            dest_path = output_dir / relative_path
            
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(file, dest_path)
            files_copied += 1
    
    print(f"✅ Converted checkpoint with {files_copied} files")
    return True

def find_latest_checkpoint(checkpoints_dir: str = "./models/checkpoints"):
    """Find the latest checkpoint in the checkpoints directory."""
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Find all checkpoint directories
    checkpoint_dirs = list(checkpoints_path.glob("checkpoint-*"))
    
    if not checkpoint_dirs:
        print("❌ No checkpoints found")
        return None
    
    # Sort by directory name (checkpoint-1000, checkpoint-2000, etc.)
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    
    latest_checkpoint = checkpoint_dirs[-1]
    print(f"✅ Found latest checkpoint: {latest_checkpoint.name}")
    
    return str(latest_checkpoint)

def main():
    """Main conversion function."""
    print("="*60)
    print("CHECKPOINT CONVERTER")
    print("="*60)
    
    # Find latest checkpoint
    checkpoints_dir = "./models/checkpoints"
    latest_checkpoint = find_latest_checkpoint(checkpoints_dir)
    
    if not latest_checkpoint:
        print("Please specify checkpoint path manually.")
        checkpoint_path = input("Enter checkpoint path: ").strip()
    else:
        print(f"\nLatest checkpoint: {latest_checkpoint}")
        use_latest = input("Use this checkpoint? (y/n): ").strip().lower()
        
        if use_latest == 'y':
            checkpoint_path = latest_checkpoint
        else:
            checkpoint_path = input("Enter checkpoint path: ").strip()
    
    # Output path
    output_path = "./models
