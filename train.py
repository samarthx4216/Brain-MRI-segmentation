"""
train.py — Train U-Net on the LGG Brain MRI Segmentation dataset.

Usage:
    python train.py --data /path/to/kaggle_3m --epochs 50 --batch 16

The best checkpoint (lowest val loss) is saved to `checkpoints/best_model.pt`.
A training log CSV is written to `checkpoints/train_log.csv`.
"""
import argparse
import os
import csv
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model   import UNet, BCEDiceLoss, dice_coefficient, iou_score
from dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net on LGG MRI dataset")
    p.add_argument("--data",       type=str,   default="./kaggle_3m",  help="Path to kaggle_3m folder")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_dir",    type=str,   default="./checkpoints")
    p.add_argument("--resume",     type=str,   default=None,           help="Resume from checkpoint")
    p.add_argument("--bilinear",   action="store_true",                 help="Use bilinear upsampling")
    p.add_argument("--bce_weight", type=float, default=0.5,            help="Weight for BCE in BCE+Dice loss")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, masks in tqdm(loader, leave=False,
                                desc="train" if train else "  val"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(preds.detach(), masks)
            total_iou  += iou_score(preds.detach(), masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        root=args.data,
        img_size=args.img_size,
        batch_size=args.batch,
        num_workers=args.workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = UNet(in_channels=3, out_channels=1, bilinear=args.bilinear).to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"Model parameters: {model.count_parameters():,}")

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val loss: {best_val_loss:.4f})")

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_file = open(log_path, "a", newline="")
    writer   = csv.writer(log_file)
    if start_epoch == 1:
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou",
                         "val_loss",   "val_dice",   "val_iou",   "lr", "time_s"])

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_dice, vl_iou = run_epoch(
            model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"Train loss={tr_loss:.4f}  dice={tr_dice:.4f}  iou={tr_iou:.4f}  |  "
            f"Val   loss={vl_loss:.4f}  dice={vl_dice:.4f}  iou={vl_iou:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )

        writer.writerow([epoch, tr_loss, tr_dice, tr_iou,
                         vl_loss, vl_dice, vl_iou, lr_now, f"{elapsed:.1f}"])
        log_file.flush()

        # ── Save best ─────────────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            ckpt_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}")

        # ── Save latest ────────────────────────────────────────────────────
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }, os.path.join(args.out_dir, "latest.pt"))

    log_file.close()
    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
