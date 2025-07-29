# Tamil-Sinhala Bidirectional Machine Translation

A comprehensive machine translation system for Tamil ↔ Sinhala translation using mBART-50 with direction-specific evaluation metrics.

## 🎯 Project Overview

This project implements a bidirectional neural machine translation system between Tamil and Sinhala languages using Facebook's mBART-50 (Multilingual BART) model. The system features advanced evaluation metrics that track translation quality separately for each direction (Tamil→Sinhala and Sinhala→Tamil).

### Key Features

- **🔄 Bidirectional Translation**: Tamil ↔ Sinhala translation support
- **📊 Direction-Specific Evaluation**: Separate metrics for each translation direction
- **🎯 Comprehensive Metrics**: BLEU, ROUGE-L, CHRF, BERTScore, Exact Match, Token Accuracy
- **🚀 GPU Optimized**: CUDA support with mixed precision training
- **📈 Real-time Monitoring**: TensorBoard integration for training visualization
- **💾 Epoch Tracking**: Detailed evaluation logs saved after each epoch
- **🏆 Best Score Tracking**: Automatic tracking of best performing epochs

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 4000 or better)
- **RAM**: Minimum 16GB (32GB recommended for large datasets)
- **Storage**: At least 10GB free space for models and datasets

### Software Requirements
```bash
# Core ML Libraries
torch >= 1.9.0
torchvision
torchaudio
transformers >= 4.20.0
datasets
accelerate

# Evaluation Metrics
evaluate
sacrebleu
rouge_score
bert_score

# Utilities
sentencepiece
protobuf
tensorboard
matplotlib
pandas
numpy
```

## 🚀 Quick Start

### 1. Environment Setup

```python
# Install PyTorch with CUDA support
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and dependencies
%pip install -U "transformers[torch]"
%pip install -U datasets accelerate evaluate sacrebleu sentencepiece protobuf rouge_score bert_score tensorboard

# Verify GPU availability
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

### 2. Data Preparation

Prepare your data in TSV format with columns:
- `source`: Source language text (Tamil or Sinhala)
- `target`: Target language text (corresponding translation)
- `source_lang`: Language code (`ta_IN` or `si_LK`)
- `target_lang`: Language code (`si_LK` or `ta_IN`)

### 3. Training

```python
# Load the training notebook
# Run: mbart50-si-ta.ipynb

# Key configuration options:
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
```

### 4. Testing

```python
# Load the testing notebook
# Run: test-si-ts.ipynb

# Test translation
sinhala_text = "ඔබට කොහොමද?"
# Expected Tamil output: "உங்களுக்கு எப்படி?"
```

## 📁 Project Structure

```
f:\S2S Research\Machine Translation si-ta\
├── README.md                          # This file
├── mbart50-si-ta.ipynb                # Main training notebook
├── test-si-ts.ipynb                   # Model testing notebook
├── data/                              # Dataset directory (create this)
│   ├── train-new.tsv                  # Training data
│   ├── val-new.tsv                    # Validation data
└── outputs_si_ta/                     # Training outputs (auto-created)
    ├── checkpoint-*/                   # Model checkpoints
    ├── epoch_*_model/                  # Epoch-specific models
    ├── direction_evaluation_epoch_*.json  # Direction-specific metrics
    └── tensorboard_logs/               # TensorBoard logs
```

## 🔧 Notebooks Description

### `mbart50-si-ta.ipynb` - Main Training Notebook

**Features:**
- **Model Loading**: mBART-50 multilingual model initialization
- **Data Processing**: TSV data loading and preprocessing for bidirectional translation
- **Training Configuration**: Optimized hyperparameters for Tamil-Sinhala translation
- **Direction-Specific Evaluation**: Advanced metrics tracking for each translation direction
- **GPU Optimization**: Mixed precision training and memory management
- **Progress Monitoring**: TensorBoard integration and evaluation logging

**Key Sections:**
1. **Environment Setup**: CUDA, PyTorch, and library installation
2. **Model & Tokenizer**: mBART-50 model and tokenizer loading
3. **Dataset Preparation**: Bidirectional data loading and preprocessing
4. **Training Configuration**: Hyperparameters and training arguments
5. **Direction-Specific Evaluation**: Enhanced metrics computation
6. **Training Execution**: Model training with progress monitoring
7. **Model Saving**: Checkpoint and final model saving

### `test-si-ts.ipynb` - Model Testing Notebook

**Features:**
- **Model Loading**: Load trained models from checkpoints
- **Interactive Testing**: Test translations with example sentences
- **Error Handling**: Robust model loading with fallback options
- **Translation Quality**: Manual evaluation of translation outputs


## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   BATCH_SIZE = 2
   GRADIENT_ACCUMULATION_STEPS = 2
   ```

2. **Model Loading Error**
   ```python
   # Check model path exists
   import os
   print(os.listdir("outputs_si_ta/"))
   ```

3. **Direction Detection Issues**
   ```python
   # Test direction detection manually
   test_preds = ["සිංහල පාසලක්", "தமிழ் பள்ளி"]
   test_refs = ["සිංහල පාසලක්", "தமிழ் பள்ளி"]
   ta_to_si, si_to_ta = detect_translation_direction(test_preds, test_refs)
   ```

### Performance Optimization

1. **GPU Memory**: Use gradient checkpointing for large models
2. **Training Speed**: Increase batch size with gradient accumulation
3. **Evaluation Speed**: Reduce BERTScore sample size during training

## 📚 References

- **mBART-50**: [Multilingual Translation with Extensible Multilingual Pretraining](https://arxiv.org/abs/2008.00401)
- **BLEU**: [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- **BERTScore**: [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- **Transformers**: [Hugging Face Transformers Library](https://huggingface.co/transformers/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test with your data
5. Submit a pull request

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

## 📞 Contact

For questions, issues, or collaborations:
- **Research Focus**: Tamil-Sinhala Neural Machine Translation
- **Technical Support**: Model training and evaluation issues
- **Data Contributions**: Additional Tamil-Sinhala parallel corpora

---

## 🌟 Acknowledgments

- **Facebook AI Research** for the mBART-50 model
- **Hugging Face** for the Transformers library
- **Tamil and Sinhala Language Communities** for linguistic resources
- **Open Source Contributors** for evaluation metrics libraries

---

*Last Updated: July 29, 2025*
*Version: 1.0.0*
