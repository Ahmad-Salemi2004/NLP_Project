# Text Summarization NLP Project

A production-ready NLP project for dialogue summarization using fine-tuned BART model on DialogSum dataset.

## Features
- Fine-tuned BART-large-CNN model on DialogSum dataset
- Dialogue summarization with state-of-the-art performance
- Complete training pipeline with validation and testing
- Interactive Jupyter notebook for experimentation
- Command-line interface for easy usage
- Comprehensive evaluation metrics (ROUGE, BLEU)
- GPU acceleration support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sadman2762/NLP_PROJECT.git
cd NLP_PROJECT/text-summarization-nlp-project

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Basic Usage
```
from src.inference import TextSummarizer

# Initialize summarizer
summarizer = TextSummarizer()

# Summarize text
dialogue = """
#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?
#Person2#: I found it would be a good idea to get a check-up.
#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.
"""

summary = summarizer.summarize(dialogue)
print(f"Summary: {summary}")
```

## Command Line Interface
```
# Train the model
python src/train.py --epochs 3 --batch_size 8

# Run inference on a file
python src/inference.py --input_file sample_dialogues.txt

# Evaluate the model
python src/evaluate.py --output_dir results/
```

## Project Structure
```
text-summarization-nlp-project/
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── evaluate.py        # Evaluation script
│   ├── utils.py           # Utility functions
│   └── data_preprocessing.py # Data processing
├── notebooks/             # Jupyter notebooks
├── models/               # Saved models
├── data/                 # Dataset storage
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
├── config.yaml          # Configuration
└── README.md           # Documentation
```
