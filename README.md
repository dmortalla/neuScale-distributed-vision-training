# neuScale — Distributed Vision Training Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Accelerate](https://img.shields.io/badge/HF-Accelerate-orange)
![GPU](https://img.shields.io/badge/Compute-Multi--GPU-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## ⚡ Tagline
Distributed PyTorch training framework for CIFAR-10 using Hugging Face Accelerate and multi-GPU optimization.

---

## 🚀 Quickstart Demo (For Reviewers)

Run a **single-epoch distributed training demo**:

```bash
pip install -r requirements.txt
accelerate config     # done once per machine
python run_demo.py
```

This verifies GPU discovery, distributed setup, loss computation, and backprop — ideal for Outlier technical reviewers.

---

## 📦 Installation (Full Training)

```bash
pip install -r requirements.txt
accelerate config
accelerate launch train_swin_cifar10.py
```

---

## 📁 Files

```text
train_swin_cifar10.py   # Full distributed training script
run_demo.py             # Lightweight single-epoch demo script
requirements.txt        # Dependencies
```

---

## 🏗 Overview

neuScale provides:

- Multi-GPU distributed training via **Hugging Face Accelerate**
- Swin Transformer (tiny) training pipeline
- Mixed precision (fp16/bf16)
- Configurable optimizers + schedulers
- Metrics, logging, and progress tracking
- Clean, modular training loop

---

## 📂 Project Structure

```text
.
├── train_swin_cifar10.py
├── run_demo.py
├── requirements.txt
├── CONTRIBUTING.md
└── .github/
    └── workflows/
        └── tests.yml
```

---

## 🧱 Architecture Overview

At a high level, the training system looks like this:

           +--------------------------+
           |      CIFAR-10 Dataset    |
           +-------------+------------+
                         |
                         v
                torchvision.datasets
                         |
                         v
           +-------------+-------------+
           |  DataLoader (shuffled,    |
           |  pinned memory, workers)  |
           +-------------+-------------+
                         |
                         v
                Accelerate.prepare(...)
                         |
                         v
           +-------------+-------------+
           |  Swin Transformer (Tiny)  |
           |  - Windowed attention     |
           |  - Patch embedding        |
           |  - Custom classification  |
           |    head for 10 classes    |
           +-------------+-------------+
                         |
                         v
               CrossEntropyLoss + AdamW
                         |
                         v
           +-------------+-------------+
           |  Mixed Precision (fp16)   |
           |  Gradient checkpointing   |
           |  Multi-GPU orchestration  |
           +---------------------------+

---

## 🤝 Contributing
See **CONTRIBUTING.md** for coding standards, branching rules, and PR workflow.

---

## 📄 License
MIT License. See `LICENSE` for details.
