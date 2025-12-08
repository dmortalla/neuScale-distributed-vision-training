# NeuScale Distributed Vision Training Framework (Swin Transformer + Accelerate)

This project trains a Swin Transformer model on CIFAR-10 using PyTorch and Hugging Face Accelerate
for easy multi-GPU / multi-process training.

It demonstrates:
- Distributed training with `accelerate`
- GPU-aware data input pipelines
- Gradient accumulation and mixed precision
- Modular training loop and clear logging

## Quickstart

```bash
pip install -r requirements.txt
accelerate config   # run once
accelerate launch train_swin_cifar10.py
```

## Files

- `train_swin_cifar10.py` — main training script (single or multi-GPU)
- `requirements.txt` — Python dependencies
