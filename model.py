"""
model.py — U-Net for brain MRI FLAIR segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two back-to-back Conv2d → BN → ReLU blocks."""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool then DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample then DoubleConv, with skip connection."""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if shapes differ
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    Args:
        in_channels:  number of input image channels (3 for RGB MRI slices)
        out_channels: number of output classes (1 for binary tumour mask)
        bilinear:     use bilinear upsampling (True) or transposed convolutions (False)
        base_features: width multiplier; default 64 matches original paper
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = True,
        base_features: int = 64,
    ):
        super().__init__()
        f = base_features
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)

        self.up1 = Up(f * 16, f * 8 // factor,  bilinear)
        self.up2 = Up(f * 8,  f * 4 // factor,  bilinear)
        self.up3 = Up(f * 4,  f * 2 // factor,  bilinear)
        self.up4 = Up(f * 2,  f,                bilinear)

        self.outc = OutConv(f, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return torch.sigmoid(self.outc(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Losses ─────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE and Dice."""
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w  = bce_weight
        self.bce    = nn.BCELoss()
        self.dice   = DiceLoss()

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + (1 - self.bce_w) * self.dice(pred, target)


# ── Metrics ────────────────────────────────────────────────────────────────────
def dice_coefficient(pred, target, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_b   = (pred > threshold).float()
    target_b = (target > threshold).float()
    inter    = (pred_b * target_b).sum()
    return ((2 * inter + smooth) / (pred_b.sum() + target_b.sum() + smooth)).item()


def iou_score(pred, target, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_b   = (pred > threshold).float()
    target_b = (target > threshold).float()
    inter    = (pred_b * target_b).sum()
    union    = pred_b.sum() + target_b.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()
