# NeuScale Distributed Vision Training Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Accelerate](https://img.shields.io/badge/HF-Accelerate-orange)
![GPU](https://img.shields.io/badge/Compute-Multi--GPU-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A high-performance distributed vision training framework built with PyTorch and Hugging Face Accelerate. Includes multi-GPU data parallelism, flexible optimizer configuration, mixed precision, and profiling hooks for scalable deep-learning experimentation.

---

## 🚀 Overview
NeuScale is a **distributed vision training system** built for CIFAR-10, showcasing real-world ML engineering techniques:

- Multi-GPU training with **Hugging Face Accelerate**
- Fine-tuning **Swin Transformer** from torchvision
- **Mixed Precision (fp16)** for speed & VRAM savings
- **Gradient checkpointing** to reduce memory footprint
- Dataloader & GPU utilization optimization
- Clean modular training loop used in production systems

This project demonstrates practical skills in scalable deep learning training, GPU performance engineering, and transformer-based visual architectures.

---

## 🚀 Quickstart Demo (For Reviewers)

Run a single-epoch distributed training demo on CIFAR-10.

```bash
pip install -r requirements.txt
accelerate config     # done once per machine
python run_demo.py
```

---

## 🧩 Features
- Distributed multi-GPU training  
- Automatic mixed precision  
- Swin Transformer backbone  
- Efficient data pipelines  
- Automatic checkpointing  
- Configurable hyperparameters  

---

## 📦 Installation

```bash
pip install -r requirements.txt
accelerate config

accelerate launch train_swin_cifar10.py

train_swin_cifar10.py   # Main distributed training script
requirements.txt         # Dependencies
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

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.
