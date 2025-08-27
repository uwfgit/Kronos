# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is a foundation model for financial candlesticks (K-lines), designed to handle the language of financial markets. It uses a two-stage framework with a specialized tokenizer and autoregressive Transformer for forecasting financial time series data.

## Key Architecture Components

### Model Structure
- **KronosTokenizer** (`model/kronos.py`): Hybrid quantization tokenizer using encoder-decoder Transformer blocks with Binary Spherical Quantization
- **Kronos** (`model/kronos.py`): Main decoder-only Transformer model for K-line prediction
- **KronosPredictor** (`model/kronos.py`): High-level interface for making predictions with data preprocessing and normalization

### Data Flow
1. Raw OHLCV (Open, High, Low, Close, Volume) data → KronosTokenizer → Discrete tokens
2. Tokenized sequences → Kronos model → Predictions
3. KronosPredictor handles the full pipeline including normalization and inverse transformation

## Common Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Predictions
```bash
python examples/prediction_example.py  # With volume data
python examples/prediction_wo_vol_example.py  # Without volume data
```

### Finetuning Pipeline
```bash
# 1. Prepare dataset (requires pyqlib)
python finetune/qlib_data_preprocess.py

# 2. Finetune tokenizer (multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py

# 3. Finetune predictor (multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py

# 4. Run backtest
python finetune/qlib_test.py --device cuda:0
```

## Important Configuration

Before finetuning, update paths in `finetune/config.py`:
- `qlib_data_path`: Path to Qlib data directory
- `dataset_path`: Directory for processed datasets
- `save_path`: Model checkpoint directory
- `pretrained_tokenizer_path`: Path to pretrained tokenizer
- `pretrained_predictor_path`: Path to pretrained model

## Model Constraints

- **Max context length**: 512 for Kronos-small and Kronos-base models
- Input data should not exceed max_context length for optimal performance
- Models available from Hugging Face Hub: NeoQuasar/Kronos-{mini,small,base}

## Key Dependencies

- PyTorch for model implementation
- Hugging Face Hub for model loading/saving
- einops for tensor operations
- pandas/numpy for data handling
- matplotlib for visualization
- pyqlib (optional) for A-share market data preparation and backtesting