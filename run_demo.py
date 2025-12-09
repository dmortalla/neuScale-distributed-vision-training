"""
Run a short demo training run for neuScale on CIFAR-10.

This is a thin wrapper around the main training script to make it obvious
for visitors how to try the project.
"""

import os
import subprocess


def main():
    # You can adjust this to match your CLI options
    cmd = ["accelerate", "launch", "train_swin_cifar10.py", "--epochs", "1"]
    print("Running demo command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
