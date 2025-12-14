#!/usr/bin/env python3
"""
Download pre-trained models for Text Summarization NLP Project
"""

import os
import sys
from pathlib import Path

def download_bart_model():
    """Download BART model from HuggingFace."""
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer
        
        print("📥 Downloading BART model from HuggingFace...")
        
        # Model name
        model_name = "facebook/bart-large-cnn"
        
        # Download model and tokenizer
        print(f"⬇️  Downloading {model_name}...")
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Save to models directory
        save_path = Path("./models/bart-base-downloaded")
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"✅ Model downloaded and saved to {save_path}")
        
        # Print model info
        print(f"\n📊 Model Information:")
        print(f"   Name: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Save location: {save_path}")
        
        return str(save_path)
        
    except ImportError:
        print("❌ Transformers library not installed.")
        print("   Run: pip install transformers")
        return None
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def download_dialogsum_dataset():
    """Download DialogSum dataset."""
    try:
        from datasets import load_dataset
        
        print("📥 Downloading DialogSum dataset...")
        
        # Download dataset
        dataset = load_dataset("knkarthick/dialogsum")
        
        print(f"✅ Dataset downloaded:")
        print(f"   Train: {len(dataset['train'])} samples")
        print(f"   Validation: {len(dataset['validation'])} samples")
        print(f"   Test: {len(dataset['test'])} samples")
        
        return dataset
        
    except ImportError:
        print("❌ Datasets library not installed.")
        print("   Run: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def main():
    """Main function to download models and datasets."""
    print("="*60)
    print("MODEL DOWNLOADER")
    print("="*60)
    
    # Create models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Menu
    print("\nWhat would you like to download?")
    print("1. BART Model (facebook/bart-large-cnn)")
    print("2. DialogSum Dataset")
    print("3. Both")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            download_bart_model()
        elif choice == "2":
            download_dialogsum_dataset()
        elif choice == "3":
            download_bart_model()
            download_dialogsum_dataset()
        elif choice == "4":
            print("👋 Exiting...")
            return
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
