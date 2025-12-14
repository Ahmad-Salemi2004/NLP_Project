# Models Directory

This directory contains all model files for the Text Summarization NLP Project.

## Directory Structure

models/
├── bart-dialogsum/ # Fine-tuned BART model (created during training)
├── base/ # Base model cache from HuggingFace
├── checkpoints/ # Training checkpoints (gitignored)
├── evaluation/ # Evaluation results and metrics
└── utils/ # Model utilities and scripts


## Model Information

### Base Model
- **Name**: `facebook/bart-large-cnn`
- **Type**: BART (Bidirectional and Auto-Regressive Transformers)
- **Purpose**: Text summarization
- **Parameters**: ~406 million
- **Size**: ~1.6 GB

### Fine-tuned Model
- **Base**: `facebook/bart-large-cnn`
- **Dataset**: DialogSum (12,460 dialogue-summary pairs)
- **Training**: 2 epochs, batch size 8, learning rate 5e-5
- **Expected Size**: ~1.6 GB
- **Location**: `models/bart-dialogsum/`

## Files in Fine-tuned Model Directory

When training completes, the following files will be created:

### Essential Files:
1. `pytorch_model.bin` - Model weights (largest file, ~1.6GB)
2. `config.json` - Model configuration
3. `tokenizer_config.json` - Tokenizer configuration
4. `vocab.json` - Vocabulary file
5. `merges.txt` - Byte-pair encoding merges

### Optional Files:
6. `generation_config.json` - Text generation settings
7. `special_tokens_map.json` - Special tokens mapping
8. `tokenizer.json` - Full tokenizer configuration
9. `model_metadata.json` - Training metadata

## How to Use

### Check if Model Exists
```python
from models import check_model_available

if check_model_available():
    print("✅ Fine-tuned model is available")
else:
    print("⚠️ Fine-tuned model not found, will use base model")
```

## Get Model Information
```
from models import get_model_info

info = get_model_info()
print(f"GPU available: {info['gpu_available']}")
print(f"Fine-tuned model: {info['fine_tuned_available']}")
```

## Load Model in App
The app.py automatically:

Checks for fine-tuned model in models/bart-dialogsum/

Falls back to base model if not found

Uses GPU if available, otherwise CPU

## Training the Model
To create the fine-tuned model:
```
# Install requirements
pip install -r requirements.txt

# Run training script
python src/train.py

# Or use the CLI command (after installation)
summarize-train --epochs 2 --batch_size 8
```
Training will create the bart-dialogsum directory with all necessary files.

## File Sizes
pytorch_model.bin: ~1.6 GB (model weights)

Other files: ~1-10 MB each

Total: ~1.7 GB

## Notes
The checkpoints/ directory is gitignored as it can be very large

Base models are cached by HuggingFace in ~/.cache/huggingface/

For production, consider model quantization to reduce size

Always verify all required files exist before deployment
