__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@example.com]"
__description__ = "Text summarization using fine-tuned BART model"
__license__ = "MIT"

# Expose main functions for easy import
from .train import train_model
from .inference import TextSummarizer, summarize_text
from .evaluate import evaluate_model
from .utils import load_config, clear_memory

__all__ = [
    "train_model",
    "TextSummarizer",
    "summarize_text",
    "evaluate_model",
    "load_config",
    "clear_memory",
    "__version__",
    "__author__",
    "__description__",
]
