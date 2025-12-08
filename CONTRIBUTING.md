# Contributing to neuScale Distributed Vision Training

Thank you for considering contributing to **neuScale**, a distributed multi-GPU vision training framework built with PyTorch and Hugging Face Accelerate.

This guide explains how to contribute code, documentation, and improvements safely and consistently.

---

## 1. Fork the Repository

Click **Fork** (top-right on GitHub) to create your own copy of this repository.

---

## 2. Clone Your Fork & Create a Branch

```bash
git clone https://github.com/<your-username>/neuScale-distributed-vision-training.git
cd neuScale-distributed-vision-training
git checkout -b feature/your-feature-name
```

---

## 3. Make Your Changes

- Keep commits small and focused.
- Follow good engineering practices (readability, modularity).
- Maintain compatibility with both **single-GPU** and **multi-GPU** setups.
- If modifying training loops, be mindful of gradient accumulation and distributed state.

---

## 4. Run Basic Checks

### Syntax Validation

```bash
python -m compileall .
```

### Optional: Run Tests (if added later)

```bash
pytest
```

### Distributed Launch Smoke Test (Recommended)

```bash
accelerate launch train_cifar_multigpu.py
```

---

## 5. Open a Pull Request

- Clearly describe the change and why it’s needed.
- Include performance benchmarks for training/throughput changes when relevant.
- Tag maintainers if modifying distributed logic or GPU performance–critical code.

---

## Code Style Guidelines

- Use Google-style docstrings.
- Prefer small, testable functions.
- Avoid unnecessary tensor allocations in critical loops.
- Document any tensor shape assumptions in the code or docstrings.

---

## Thank You

Your contributions help improve distributed deep-learning tooling and make the project more robust for everyone.

