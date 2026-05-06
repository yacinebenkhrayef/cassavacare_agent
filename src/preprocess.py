import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────
def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.1, hue=0.03),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Same as val_transforms — use for single-image inference."""
    return get_val_transforms(img_size)


# ── Denormalise (for visualisation) ──────────────────────────
def denorm(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return HWC numpy array in [0, 1]."""
    t = tensor.clone()
    for ch, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[ch] = t[ch] * s + m
    return t.permute(1, 2, 0).numpy().clip(0, 1)


# ── Dataset ──────────────────────────────────────────────────
class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["label"], dtype=torch.long)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_loaders(cfg: dict, train_df, val_df, test_df) -> tuple:
    size = cfg["img_size"]
    train_tf = get_train_transforms(size)
    val_tf = get_val_transforms(size)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=cfg["pin_memory"], persistent_workers=cfg["persistent_workers"])
    train_loader = DataLoader(
        CassavaDataset(train_df, cfg["base_img_dir"], train_tf),
        shuffle=True, worker_init_fn=seed_worker, generator=g, **kw)
    val_loader = DataLoader(
        CassavaDataset(val_df, cfg["base_img_dir"], val_tf),
        shuffle=False, **kw)
    test_loader = DataLoader(
        CassavaDataset(test_df, cfg["base_img_dir"], val_tf),
        shuffle=False, **kw)
    return train_loader, val_loader, test_loader
