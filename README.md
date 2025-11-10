# Mini Transformer + LoRA Fine-Tuning Framework

This project implements a compact Transformer architecture with LoRA fine-tuning and SentencePiece tokenization.  
It is fully self-contained and requires no external frameworks such as Hugging Face Transformers or Axolotl.

---

## Overview

The repository provides a **trainable, minimal GPT-like architecture** with the following features:

- Decoder-only Transformer built entirely from scratch in PyTorch  
- SentencePiece or character-level tokenizer  
- LoRA parameter-efficient fine-tuning  
- Mixed precision (AMP) and EMA support  
- Cosine or cosine-restart learning rate schedule with warmup  
- Full checkpoint resume and evaluation  
- Independent sampling script for text generation  

This framework serves both as an educational reference and a practical minimal fine-tuning system for small LMs.

---

## Directory Structure

```
project_root/
├── attention.py       # Scaled dot-product & multi-head attention (supports GQA)
├── model.py           # Core Transformer blocks & DecoderOnlyLM
├── data.py            # Tokenizer (SPM/Char) and dataset utilities
├── lora.py            # LoRA injection, merging, and adapter save/load
├── train.py           # Main training loop with EMA, scheduler, and SFT mode
├── quick_sample.py    # Lightweight inference / generation script
├── data/
│   ├── spm_en16k.model    # SentencePiece model
│   ├── spm_en16k.vocab
│   └── your_dataset.txt   # Training text
└── pth/
    ├── ckpt_minGPT.pth    # Training checkpoints
    └── ckpt_best.pth
```

---

## Training

### Pretraining (Language Modeling)

```bash
python train.py \
  DATA_PATH=./data/your_dataset.txt \
  SPM_MODEL=./data/spm_en16k.model \
  CKPT_PATH=./pth/run_lm_en.pth \
  MAX_STEPS=10000 \
  D_MODEL=384 N_LAYER=6 N_HEAD=6
```

### Fine-tuning with LoRA

```bash
export USE_LORA=1
export BASE_CKPT=./pth/run_lm_en.pth
python train.py \
  DATA_PATH=./data/sft_data.txt \
  CKPT_PATH=./pth/run_sft_en_final.pth \
  SPM_MODEL=./data/spm_en16k.model
```

Environment variables can control most settings:

| Variable | Description | Default |
|-----------|--------------|----------|
| `USE_LORA` | Enable LoRA fine-tuning | `0` |
| `SFT_MODE` | Enable SFT mask (learn only after "### Response:") | `0` |
| `EMA` | Apply Exponential Moving Average | `1` |
| `MAX_STEPS` | Training steps | `10000` |
| `LR` | Learning rate | `3e-4` |
| `SEQ_LEN` | Context length | `256` |
| `COSINE_RESTARTS` | Use cosine restart schedule | `0` |

---

## Checkpoints

All checkpoints are saved in `pth/` and contain model weights, optimizer state, and training metadata.

- `ckpt_minGPT.pth` – in-progress checkpoint  
- `ckpt_best.pth` – best validation loss checkpoint  

Each checkpoint includes the config dict, allowing direct reload for inference.

---

## Inference

Generate text using the standalone sampling script:

```bash
python quick_sample.py \
  --ckpt ./pth/ckpt_best.pth \
  --spm ./data/spm_en16k.model \
  --prompt "The future of AI research is"
```

Arguments:

| Flag | Description | Default |
|------|--------------|----------|
| `--temp` | Sampling temperature | `0.6` |
| `--top_p` | Nucleus sampling cutoff | `0.92` |
| `--min_new`, `--max_new` | Min/max generated tokens | 40 / 120 |
| `--rep_penalty` | Repetition penalty | 1.2 |
| `--no_repeat` | Ban last N tokens | 5 |

Example output:
```
### Prompt 1
The future of AI research is
---
The future of AI research lies in small, specialized models that can be trained locally with efficient fine-tuning.
```

---

## Implementation Details

| Component | File | Description |
|------------|------|-------------|
| Attention | `attention.py` | Multi-head & grouped-query attention using PyTorch SDPA |
| Model | `model.py` | Decoder-only Transformer with parallel residual (SwiGLU + MHA) |
| LoRA | `lora.py` | Injects `LoRALinear` layers, supports merge/unmerge and adapter saving |
| Data | `data.py` | SentencePiece tokenizer and efficient data slicing (memmap ready) |
| Training | `train.py` | Full LM & SFT loop with checkpointing, warmup-cosine LR, EMA |
| Sampling | `quick_sample.py` | Minimal temperature/top-p text generation utility |

---

## Example Workflow

1. **Prepare Dataset:** Place your raw text under `data/`.  
2. **Train Base Model:** Run `train.py` for LM pretraining.  
3. **Fine-tune (Optional):** Set `USE_LORA=1` and run LoRA fine-tuning.  
4. **Run Inference:** Use `quick_sample.py` to generate outputs.  

---

## Future Extensions

- Add DPO / PPO fine-tuning integration  
- Convert weights for Hugging Face compatibility  
- Quantized `.gguf` export for llama.cpp inference  
- Multi-GPU or distributed training  
