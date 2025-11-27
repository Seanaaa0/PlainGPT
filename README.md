# PlainGPT: A Lightweight, From-Scratch Transformer for Mechanistic Study & PEFT Verification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-blue?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Abstract

**PlainGPT** is a minimalist, decoder-only Transformer model implemented entirely from scratch in PyTorch, designed to run under constrained local compute resources.

Unlike projects that rely on high-level APIs (e.g., HuggingFace `transformers` or `peft`), PlainGPT manually implements the mathematical foundations of the **Self-Attention mechanism**, **Positional Encodings**, and **Low-Rank Adaptation (LoRA)**. The primary goal of this project is not to compete with commercial LLMs in dialogue quality, but to serve as a **transparent testbed** for understanding:
1.  Gradient flow in causal attention masking.
2.  The convergence behavior of LoRA in small-scale models (~30M parameters).
3.  Syntactic pattern acquisition in specialized datasets (TinyShakespeare).

---

## Key Features

* ** Implementation From Scratch**:
    * Manual implementation of **Scaled Dot-Product Attention** (following *Vaswani et al., 2017*).
    * Custom **Sinusoidal Positional Embeddings**.
    * Hand-coded **Causal Masking** to ensure strict autoregressive properties.
* ** Parameter-Efficient Fine-Tuning (PEFT)**:
    * Integrated **LoRA (Low-Rank Adaptation)** by manually injecting low-rank matrices ($A \times B$) into linear projection layers.
    * Supports freezing backbone weights to study efficient adaptation dynamics.
* ** Optimized for Interpretability**:
    * Clean, modular code structure designed for inspecting attention weights and tensor shapes at every layer.

---

##  Model Architecture

Due to local hardware constraints, PlainGPT adopts a scaled-down architecture optimized for rapid iteration and convergence verification.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | Decoder-only | GPT-style autoregressive model |
| **Parameters** | ~30M | Lightweight design for consumer GPUs |
| **Context Window** | 256 / 512 | Adjusted for memory efficiency |
| **Embedding Dim** | 128 / 384 | Reduced dimension ($d_{model}$) |
| **Heads** | 4 / 6 | Maintains $d_k = 64$ for representation capacity |
| **Layers** | 4 / 6 | Reduced depth for faster backward pass |

### Attention Mechanism Implementation
The core attention logic is implemented as a direct translation of the mathematical definition:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

*(See `model.py` for the explicit PyTorch implementation involving tensor transpositions and causal masking.)*

---

##  Experimental Results

### Training Convergence
The model was trained on the **TinyShakespeare** dataset to test its ability to capture syntactic structures and Early Modern English vocabulary.

> **[Insert Your Loss Curve Image Here]**
>
> *Figure 1: Training loss over epochs. The downward trend confirms the correct implementation of the backpropagation pipeline and attention gradients.*

### Inference & Observations
The model successfully learned the vocabulary and sentence structure of the training data.

**Sample Output (Uncurated):**
> *[Insert a short example of the generated text here, e.g., "To be or not to be, that is the question..."]*

**Analysis of Artifacts:**
While the model captures the *style* of Shakespeare, users may observe repetitive punctuation or local incoherence in long sequences. These artifacts are attributed to:
1.  **Limited Model Capacity:** At ~30M parameters, the model prioritizes local syntactic patterns (n-grams) over long-range semantic dependencies.
2.  **Greedy Decoding:** The current inference uses basic sampling; implementing Top-k/Nucleus sampling would improve diversity.
3.  **Data Bias:** The TinyShakespeare dataset contains idiosyncratic punctuation patterns that the model overfits to.

---

##  Usage

To inspect the model architecture or run training:

```bash
# Clone the repository
git clone [https://github.com/Seanaaa0/PlainGPT.git](https://github.com/Seanaaa0/PlainGPT.git)
cd PlainGPT

# Install dependencies
pip install torch numpy

# Train the model (from scratch)
python train.py

# Run inference
python generate.py


---

##  References

This project is built upon the foundational concepts introduced in:
1. Attention Is All You Need (Vaswani et al., 2017) - Base Architecture
2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) - Fine-tuning Strategy
3. GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019) - Decoder-only Structure
