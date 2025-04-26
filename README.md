# DualHeadBERT - Toxicity and Identity-Based Threat Detection

## Overview
This repository contains training and evaluation scripts for a **DualHeadBERT** model designed to detect general toxicity and identity-based threats in text data.  
The model architecture is based on BERT with two separate heads to handle **normal** and **imbalanced** label groups.

## Directory Structure
- `ezgi_train.py`:  
  Initial training script using a standard `torch.nn.Module`-based DualHeadBERT model.
- `ezgi_train_new.py`:  
  Updated training script where DualHeadBERT inherits from `BertPreTrainedModel`, enabling compatibility with Hugging Face’s `Trainer` class and standardized checkpoint saving.
- `ezgi_test.py`:  
  Evaluation script for models trained with the original (`nn.Module`-based) architecture, manually loading `.safetensors` files.
- `ezgi_test_new.py`:  
  Evaluation script for models trained with the Hugging Face-compatible DualHeadBERT, allowing direct loading via `from_pretrained`.

## Installation
Before running the scripts, install the required packages:

```bash
pip install torch transformers datasets scikit-multilearn scikit-learn safetensors
```

## Usage

### Training
Train two models (for normal and identity labels):

```bash
python ezgi_train_new.py
```
- Saves models under `saved_model_n/` and `saved_model_i/`
- Saves the tokenizer under `saved_tokenizer/`

Alternatively, for the initial version:

```bash
python ezgi_train.py
```

### Testing and Evaluation
Evaluate the models on the stratified test set:

```bash
python ezgi_test_new.py
```
- Automatically loads models from `saved_model_n/` and `saved_model_i/`
- Computes precision, recall, F1 scores per label
- Plots precision-recall curves for both normal and identity labels.

Alternatively, for the initial version:

```bash
python ezgi_test.py
```
- Manually loads model weights from `.safetensors` files.
- Performs grid search over different gamma and alpha values for Focal Loss.
- 
## Resources

- [Download Trained Models, Tokenizer, and Dataset (saved_model_n/, saved_model_i/, and saved_tokenizer/)](https://drive.google.com/drive/folders/179o12QLpj2XlzKGcGkmqfd0RMX-mCX0y?usp=drive_link)



## Model Architecture
- **Backbone:** BERT (bert-base-uncased)
- **Dual Heads:**
  - `head_n`: General toxicity labels (threat, insult, obscene, identity attack)
  - `head_i`: Identity-related labels (e.g., black, muslim, female)
- **Loss Functions:**
  - `head_n`: BCEWithLogitsLoss with class balancing (`pos_weight`)
  - `head_i`: Focal Loss to handle severe class imbalance

## Notes
- **Class Imbalance Handling:**  
  Identity labels often have very low positive rates (<0.1%). Focal Loss and per-class `pos_weight` strategies were applied.
- **Threshold Optimization:**  
  Validation-based threshold search was conducted per class to maximize F1-score instead of using a fixed 0.5 threshold.
- **Mixed Precision Training:**  
  Scripts use FP16 precision automatically if a compatible GPU is available.

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.15+
- scikit-learn
- scikit-multilearn
- safetensors

## Authors
- Sena Ezgi Anadollu

## License
This project is for academic and research purposes.
