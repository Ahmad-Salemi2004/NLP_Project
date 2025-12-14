import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data loading and processing for the summarization project."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary data directories."""
        directories = [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.data_dir / "examples",
            self.data_dir / "splits",
            self.data_dir / "statistics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def load_dialogsum(self, split: str = "train", cache: bool = True):
        """Load DialogSum dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            logger.info(f"📦 Loading DialogSum {split} split...")
            
            # Load dataset
            dataset = load_dataset(
                "knkarthick/dialogsum",
                split=split,
                cache_dir=str(self.cache_dir)
            )
            
            logger.info(f"✅ Loaded {len(dataset)} samples from DialogSum {split}")
            return dataset
            
        except ImportError:
            logger.error("❌ datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load DialogSum: {e}")
            raise
    
    def save_examples(self, examples: List[Dict], filename: str = "dialogues.json"):
        """Save example dialogues to JSON file."""
        examples_path = self.data_dir / "examples" / filename
        
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(examples)} examples to {examples_path}")
        return str(examples_path)
    
    def load_examples(self, filename: str = "dialogues.json") -> List[Dict]:
        """Load example dialogues from JSON file."""
        examples_path = self.data_dir / "examples" / filename
        
        if not examples_path.exists():
            logger.warning(f"⚠️ Examples file not found: {examples_path}")
            return self.get_default_examples()
        
        with open(examples_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        logger.info(f"📖 Loaded {len(examples)} examples from {examples_path}")
        return examples
    
    def get_default_examples(self) -> List[Dict]:
        """Return default example dialogues if file doesn't exist."""
        return [
            {
                "id": 1,
                "title": "Doctor Appointment",
                "category": "Healthcare",
                "dialogue": "#Person1#: Hi, I'm Doctor Smith. How can I help you today?\n#Person2#: I've been having headaches for the past week.\n#Person1#: Let me check your blood pressure first."
            },
            {
                "id": 2,
                "title": "Job Interview",
                "category": "Employment",
                "dialogue": "#Person1#: Tell me about your experience.\n#Person2#: I worked as a developer for 2 years.\n#Person1#: What programming languages do you know?"
            }
        ]
    
    def analyze_dataset(self, dataset) -> Dict:
        """Analyze dataset and return statistics."""
        logger.info("📊 Analyzing dataset statistics...")
        
        # Get basic statistics
        stats = {
            "num_samples": len(dataset),
            "dialogue_lengths": [],
            "summary_lengths": [],
            "compression_ratios": []
        }
        
        # Calculate lengths
        for example in dataset:
            dialogue_words = len(example["dialogue"].split())
            summary_words = len(example["summary"].split())
            
            stats["dialogue_lengths"].append(dialogue_words)
            stats["summary_lengths"].append(summary_words)
            
            if summary_words > 0:
                stats["compression_ratios"].append(dialogue_words / summary_words)
        
        # Calculate summary statistics
        import numpy as np
        
        stats["dialogue_stats"] = {
            "mean": np.mean(stats["dialogue_lengths"]),
            "median": np.median(stats["dialogue_lengths"]),
            "min": np.min(stats["dialogue_lengths"]),
            "max": np.max(stats["dialogue_lengths"]),
            "std": np.std(stats["dialogue_lengths"])
        }
        
        stats["summary_stats"] = {
            "mean": np.mean(stats["summary_lengths"]),
            "median": np.median(stats["summary_lengths"]),
            "min": np.min(stats["summary_lengths"]),
            "max": np.max(stats["summary_lengths"]),
            "std": np.std(stats["summary_lengths"])
        }
        
        stats["compression_stats"] = {
            "mean": np.mean(stats["compression_ratios"]),
            "median": np.median(stats["compression_ratios"]),
            "min": np.min(stats["compression_ratios"]),
            "max": np.max(stats["compression_ratios"])
        }
        
        logger.info("✅ Dataset analysis complete")
        return stats
    
    def save_statistics(self, stats: Dict, filename: str = "dataset_stats.json"):
        """Save dataset statistics to JSON file."""
        stats_path = self.data_dir / "statistics" / filename
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📈 Saved statistics to {stats_path}")
        return str(stats_path)

# Create singleton instance
data_manager = DataManager()

# Convenience functions
def load_dataset(split="train"):
    """Load DialogSum dataset."""
    return data_manager.load_dialogsum(split)

def get_examples():
    """Get example dialogues."""
    return data_manager.load_examples()

def analyze_data(dataset):
    """Analyze dataset statistics."""
    return data_manager.analyze_dataset(dataset)
