#!/usr/bin/env python3

import torch
import os
import json
from pathlib import Path
from typing import Dict, Optional

def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (GPU not available)")
    
    return device

def check_model_files(model_path: str) -> bool:
    """Check if all required model files exist."""
    required_files = [
        "pytorch_model.bin",
        "config.json", 
        "tokenizer_config.json"
    ]
    
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {missing_files}")
        return False
    
    print(f"✅ All model files found in {model_path}")
    return True

def get_model_size(model_path: str) -> str:
    """Calculate model size in human-readable format."""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        return "Directory not found"
    
    total_size = 0
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    # Convert to appropriate unit
    if total_size < 1024:
        return f"{total_size} B"
    elif total_size < 1024 * 1024:
        return f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        return f"{total_size / (1024 * 1024):.1f} MB"
    else:
        return f"{total_size / (1024 * 1024 * 1024):.1f} GB"

def create_model_metadata(training_config: Dict) -> Dict:
    """Create metadata for the trained model."""
    metadata = {
        "model": {
            "base_model": "facebook/bart-large-cnn",
            "fine_tuned_on": "DialogSum",
            "purpose": "Dialogue summarization"
        },
        "training": {
            "epochs": training_config.get("epochs", 2),
            "batch_size": training_config.get("batch_size", 8),
            "learning_rate": training_config.get("learning_rate", 5e-5),
            "dataset_size": training_config.get("dataset_size", 12460)
        },
        "hardware": {
            "gpu_used": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "performance": {
            "final_loss": training_config.get("final_loss", 0.0),
            "training_time": training_config.get("training_time", "N/A")
        },
        "creation_date": "2024-01-15",
        "created_by": "NLP Text Summarization Project",
        "notes": "Fine-tuned for college NLP course project"
    }
    
    return metadata

def save_model_metadata(model_path: str, metadata: Dict):
    """Save model metadata to JSON file."""
    metadata_file = Path(model_path) / "model_metadata.json"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved model metadata to {metadata_file}")

def load_model_config(model_path: str) -> Optional[Dict]:
    """Load model configuration from JSON file."""
    config_file = Path(model_path) / "config.json"
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def print_model_info(model_path: str):
    """Print information about the model."""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Get basic info
    model_size = get_model_size(model_path)
    config = load_model_config(model_path)
    
    print(f"📍 Path: {model_path}")
    print(f"📦 Size: {model_size}")
    
    if config:
        print(f"🤖 Type: {config.get('model_type', 'Unknown')}")
        print(f"🔢 Parameters: {config.get('vocab_size', 'Unknown')} vocabulary")
        print(f"📐 Layers: {config.get('num_hidden_layers', 'Unknown')}")
    
    # Check for metadata
    metadata_file = Path(model_path) / "model_metadata.json"
    if metadata_file.exists():
        print("📄 Metadata: Available")
    
    print("="*60)

if __name__ == "__main__":
    # Example usage
    device = get_device()
    print(f"Using device: {device}")
    
    # Check a model path
    model_path = "./models/bart-dialogsum"
    if check_model_files(model_path):
        print_model_info(model_path)
