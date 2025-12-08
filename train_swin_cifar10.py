"""NeuScale Distributed Vision Training: Swin Transformer on CIFAR-10."""

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from accelerate import Accelerator
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    num_epochs: int = 5
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_classes: int = 10
    log_interval: int = 50


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for CIFAR-10."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=transform_train
    )
    test_ds = datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_model(num_classes: int) -> nn.Module:
    """Build a Swin Transformer model using torchvision."""
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    cfg: TrainConfig,
) -> None:
    """Train the model for a single epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}"
    )

    for step, (images, targets) in enumerate(progress_bar, start=1):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        if step % cfg.log_interval == 0:
            avg_loss = total_loss / cfg.log_interval
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            total_loss = 0.0


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    """Evaluate model on the validation set and return accuracy."""
    model.eval()
    correct, total = 0, 0
    for images, targets in dataloader:
        outputs = model(images)
        _ = loss_fn(outputs, targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = correct / total
    if accelerator.is_local_main_process:
        print(f"Validation accuracy: {acc:.4f}")
    return acc


def main() -> None:
    cfg = TrainConfig()
    accelerator = Accelerator(mixed_precision="fp16")

    if accelerator.is_local_main_process:
        os.makedirs("checkpoints", exist_ok=True)
        print("Using device:", accelerator.device)

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg.num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    best_acc = 0.0
    for epoch in range(1, cfg.num_epochs + 1):
        train_one_epoch(accelerator, model, train_loader, optimizer, loss_fn, epoch, cfg)
        acc = evaluate(accelerator, model, test_loader, loss_fn)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process and acc > best_acc:
            best_acc = acc
            accelerator.print(f"New best accuracy: {best_acc:.4f}, saving checkpoint.")
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), f"checkpoints/swin_cifar10_best.pt")


if __name__ == "__main__":
    main()
