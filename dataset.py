"""
dataset.py — LGG Brain MRI Segmentation Dataset loader
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class LGGDataset(Dataset):
    """
    LGG Segmentation dataset from Kaggle / TCIA.

    Directory layout expected:
        root/
          TCGA_CS_xxxx_xxxxxxxx/
            <name>_<slice>.tif          ← MRI slice
            <name>_<slice>_mask.tif     ← Corresponding binary mask

    Args:
        root        : path to `kaggle_3m/` folder
        img_size    : square resize target (default 256)
        augment     : apply random augmentations when True
        transform   : optional custom transform for images
        split       : 'train' | 'val' | 'all'
        val_split   : fraction of data held out for validation
        seed        : random seed for reproducible split
    """

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        augment: bool = False,
        transform: Optional[Callable] = None,
        split: str = "all",
        val_split: float = 0.15,
        seed: int = 42,
    ):
        self.img_size  = img_size
        self.augment   = augment
        self.transform = transform

        # Collect all (image, mask) pairs
        mask_paths = sorted(glob.glob(os.path.join(root, "**", "*_mask.tif"), recursive=True))
        pairs = []
        for mask_path in mask_paths:
            img_path = mask_path.replace("_mask.tif", ".tif")
            if os.path.exists(img_path):
                pairs.append((img_path, mask_path))

        if not pairs:
            raise FileNotFoundError(
                f"No image-mask pairs found under {root!r}. "
                "Ensure the dataset is downloaded and extracted correctly."
            )

        # Train / val split
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_split))

        if split == "train":
            self.pairs = pairs[n_val:]
        elif split == "val":
            self.pairs = pairs[:n_val]
        else:
            self.pairs = pairs

        print(f"[LGGDataset] split={split!r}  samples={len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(path))
        return img

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        """Paired augmentation ensuring image and mask get identical transforms."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        # Random rotation ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)
        # Random brightness / contrast jitter (image only)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # ── Load image ────────────────────────────────────────────────────
        img = self._load(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # ── Load mask ─────────────────────────────────────────────────────
        mask = self._load(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        # ── To tensors ────────────────────────────────────────────────────
        img_t  = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)       # (1, H, W)

        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t, mask_t

    @property
    def positive_ratio(self) -> float:
        """Fraction of slices that have a non-empty mask (tumour present)."""
        count = 0
        for _, mask_path in self.pairs:
            mask = self._load(mask_path)
            if mask.max() > 0:
                count += 1
        return count / len(self.pairs)


def get_dataloaders(
    root: str,
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) ready for training."""
    train_ds = LGGDataset(root, img_size=img_size, augment=True,
                          split="train", val_split=val_split, seed=seed)
    val_ds   = LGGDataset(root, img_size=img_size, augment=False,
                          split="val",   val_split=val_split, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader
