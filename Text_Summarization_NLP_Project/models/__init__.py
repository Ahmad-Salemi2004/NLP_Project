import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading, saving, and utilities."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.fine_tuned_dir = self.models_dir / "bart-dialogsum"
        self.base_dir = self.models_dir / "base"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.evaluation_dir = self.models_dir / "evaluation"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create model directories if they don't exist."""
        directories = [
            self.models_dir,
            self.fine_tuned_dir,
            self.base_dir,
            self.checkpoints_dir,
            self.evaluation_dir,
            self.models_dir / "utils"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created model directory: {directory}")
    
    def check_fine_tuned_model(self) -> bool:
        """Check if fine-tuned model exists."""
        required_files = [
            "pytorch_model.bin",
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        
        if not self.fine_tuned_dir.exists():
            logger.warning("⚠️ Fine-tuned model directory not found")
            return False
        
        # Check for required files
        missing_files = []
        for file in required_files:
            if not (self.fine_tuned_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠️ Missing model files: {missing_files}")
            return False
        
        logger.info("✅ Fine-tuned model found and complete")
        return True
    
    def get_model_info(self) -> Dict:
        """Get information about available models."""
        info = {
            "fine_tuned_available": self.check_fine_tuned_model(),
            "gpu_available": torch.cuda.is_available(),
            "model_paths": {},
            "model_sizes": {}
        }
        
        # Get fine-tuned model info
        if info["fine_tuned_available"]:
            info["model_paths"]["fine_tuned"] = str(self.fine_tuned_dir)
            info["model_sizes"]["fine_tuned"] = self._get_directory_size(self.fine_tuned_dir)
        
        # Base model info
        info["model_paths"]["base"] = "facebook/bart-large-cnn (HuggingFace Hub)"
        
        # Checkpoint info
        if self.checkpoints_dir.exists():
            checkpoints = list(self.checkpoints_dir.glob("checkpoint-*"))
            info["checkpoints_available"] = len(checkpoints)
            info["checkpoints"] = [str(c.name) for c in checkpoints[:5]]  # First 5
        
        return info
    
    def _get_directory_size(self, directory: Path) -> str:
        """Calculate directory size in MB."""
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    
    def save_model_metadata(self, metadata: Dict):
        """Save model metadata."""
        metadata_file = self.fine_tuned_dir / "model_metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved model metadata to {metadata_file}")
    
    def load_model_metadata(self) -> Optional[Dict]:
        """Load model metadata."""
        metadata_file = self.fine_tuned_dir / "model_metadata.json"
        
        if not metadata_file.exists():
            logger.warning("⚠️ Model metadata not found")
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"📖 Loaded model metadata from {metadata_file}")
        return metadata
    
    def save_evaluation_results(self, results: Dict, filename: str = "metrics.json"):
        """Save evaluation results."""
        results_file = self.evaluation_dir / filename
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Saved evaluation results to {results_file}")
        return str(results_file)
    
    def save_training_log(self, log: str, filename: str = "training_logs.txt"):
        """Save training logs."""
        log_file = self.evaluation_dir / filename
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Log entry: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(log)
        
        logger.info(f"📝 Saved training log to {log_file}")
        return str(log_file)

# Import datetime for logging
from datetime import datetime

# Create singleton instance
model_manager = ModelManager()

# Convenience functions
def check_model_available() -> bool:
    """Check if fine-tuned model is available."""
    return model_manager.check_fine_tuned_model()

def get_model_info() -> Dict:
    """Get information about models."""
    return model_manager.get_model_info()

def save_metadata(metadata: Dict):
    """Save model metadata."""
    return model_manager.save_model_metadata(metadata)
