# Text Summarization NLP Project

A production-ready NLP project for dialogue summarization using fine-tuned BART model on DialogSum dataset.

## 📋 Features
- Fine-tuned BART-large-CNN model on DialogSum dataset
- Dialogue summarization with state-of-the-art performance
- Complete training pipeline with validation and testing
- Interactive Jupyter notebook for experimentation
- Command-line interface for easy usage
- Comprehensive evaluation metrics (ROUGE, BLEU)
- GPU acceleration support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sadman2762/NLP_PROJECT.git
cd NLP_PROJECT/text-summarization-nlp-project

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .


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
