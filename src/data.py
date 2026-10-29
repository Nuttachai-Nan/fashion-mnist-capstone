"""Data loading with a deterministic validation split."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.2860,)
STD = (0.3530,)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations make runs more repeatable, sometimes at a speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(data_dir: str = "data", batch_size: int = 128, seed: int = 42,
                 num_workers: int = 2):
    """Return train/validation/test loaders; test never participates in tuning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    n_val = 5000
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_full, [n_train, n_val], generator=generator
    )
    # Distinct loader generator makes shuffle reproducible without altering global RNG.
    loader_generator = torch.Generator().manual_seed(seed)
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory,
                              generator=loader_generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory)
    return train_loader, val_loader, test_loader, test_set
