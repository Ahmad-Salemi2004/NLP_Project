# Data Directory

This directory contains all data files for the Text Summarization NLP Project.

## Directory Structure
data/
├── raw/ # Raw datasets (gitignored)
├── processed/ # Processed datasets in JSONL format
├── cache/ # HuggingFace cache (gitignored)
├── examples/ # Example dialogues for testing
├── splits/ # Dataset split information
├── statistics/ # Dataset statistics and visualizations
└── README.md # This file


## Dataset Information

### DialogSum Dataset
The project uses the **DialogSum** dataset from HuggingFace:
- **Source**: `knkarthick/dialogsum`
- **Size**: 12,460 training samples, 500 validation, 1,500 test
- **Format**: Each sample contains:
  - `dialogue`: Multi-turn conversation
  - `summary`: Concise summary
  - `topic`: Conversation topic

### Dataset Statistics
- Average dialogue length: 127 words
- Average summary length: 21 words  
- Average compression ratio: 6.2x
- Topics: Healthcare, Employment, Travel, Food, Education, etc.

## Usage

### Loading Data
```python
from data import load_dataset, get_examples

# Load DialogSum dataset
train_data = load_dataset("train")
val_data = load_dataset("validation")

# Get example dialogues
examples = get_examples()
```

## Data Processing
The data is pre-processed and saved in JSONL format:

train.jsonl: Training data

validation.jsonl: Validation data

test.jsonl: Test data

## Example Dialogues

Example dialogues are stored in examples/ directory:

dialogues.json: Main examples for the web interface

healthcare.json: Healthcare-related examples

employment.json: Employment-related examples

travel.json: Travel-related examples

## File Formats
JSONL Format (JSON Lines)
Each line is a valid JSON object:
```
{"id": "train_0", "dialogue": "...", "summary": "...", "topic": "..."}
```

## Statistics Files
dataset_stats.json: Comprehensive dataset statistics

length_distribution.py: Script to generate visualization

## Git Ignored Files
The following directories are gitignored:

raw/: Contains original raw data files

cache/: Contains
