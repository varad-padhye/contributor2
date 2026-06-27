"""
train.py — NCM training & evaluation entry point.

Usage
-----
    python train.py --config config.yaml [--overrides key=value ...]

Modes (set via config.yaml → training.mode):
  "ncm"     — plain NCM: one full pass to compute means, no gradient step.
  "ncm_ml"  — NCM + metric learning (W) via SGD on the NCM cross-entropy loss.
  "ncm_mm"  — NCM-MM: sub-means via EM + optional W.

Backbone
  Any torchvision model name, e.g. "resnet50".
  Set backbone.pretrained=true to load ImageNet weights.
  All backbone parameters are frozen throughout.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models

from model import FrozenBackbone, NCMClassifier

# ---------------------------------------------------------------------------
log = logging.getLogger("ncm")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: Dict, overrides: list[str]) -> None:
    """Apply key=value overrides (dot-separated keys) in-place."""
    for item in overrides:
        k, v = item.split("=", 1)
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d[key]
        # naive cast
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        d[keys[-1]] = v


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(cfg: Dict) -> tuple[T.Compose, T.Compose]:
    img_size = cfg["data"].get("image_size", 224)
    mean = cfg["data"].get("mean", [0.485, 0.456, 0.406])
    std  = cfg["data"].get("std",  [0.229, 0.224, 0.225])

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_loaders(cfg: Dict) -> tuple[DataLoader, DataLoader]:
    train_tf, val_tf = build_transforms(cfg)
    data_root = cfg["data"]["root"]

    dataset_name = cfg["data"].get("dataset", "imagefolder").lower()

    if dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(data_root, train=True,  transform=train_tf, download=True)
        val_ds   = datasets.CIFAR100(data_root, train=False, transform=val_tf,   download=True)
    elif dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(data_root, train=True,  transform=train_tf, download=True)
        val_ds   = datasets.CIFAR10(data_root, train=False, transform=val_tf,   download=True)
    else:
        # Generic ImageFolder layout: data_root/train/  data_root/val/
        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_tf)

    num_workers = cfg["data"].get("num_workers", 4)
    batch_size  = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    num_classes = len(train_ds.classes) if hasattr(train_ds, "classes") else cfg["model"]["num_classes"]
    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def build_backbone(cfg: Dict, device: torch.device) -> tuple[nn.Module | None, int]:
    """
    Returns (backbone, feat_dim).
    backbone is None when mode=='ncm' and no backbone is configured.
    """
    bb_cfg = cfg.get("backbone", {})
    name = bb_cfg.get("name", None)
    if name is None:
        # Raw features expected from the dataset (pre-extracted)
        return None, cfg["model"]["feat_dim"]

    pretrained = bb_cfg.get("pretrained", True)
    weights_arg = "DEFAULT" if pretrained else None

    model_fn = getattr(models, name, None)
    if model_fn is None:
        raise ValueError(f"Unknown torchvision model: {name}")

    backbone = model_fn(weights=weights_arg)

    # Strip the final classification head; record output dim
    if hasattr(backbone, "fc"):
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        # VGG / MobileNet / EfficientNet style
        last = list(backbone.classifier.children())[-1]
        feat_dim = last.in_features
        backbone.classifier[-1] = nn.Identity()
    elif hasattr(backbone, "head"):
        feat_dim = backbone.head.in_features
        backbone.head = nn.Identity()
    else:
        raise RuntimeError("Cannot infer feat_dim from this backbone architecture.")

    backbone = FrozenBackbone(backbone).to(device)
    backbone.eval()
    return backbone, feat_dim


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: Path, model: NCMClassifier, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    log.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: NCMClassifier,
    loader: DataLoader,
    backbone: nn.Module | None,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total = 0

    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)

        if backbone is not None:
            feats = backbone(imgs)
        else:
            feats = imgs

        logits = model(feats)
        loss   = nn.functional.cross_entropy(logits, labels)
        preds  = logits.argmax(dim=-1)

        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)

    return total_loss / total, total_correct / total


# ---------------------------------------------------------------------------
# Mode: plain NCM (no gradient — one pass mean computation)
# ---------------------------------------------------------------------------

def run_ncm(
    model: NCMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    backbone: nn.Module | None,
    device: torch.device,
    cfg: Dict,
    ckpt_dir: Path,
) -> None:
    log.info("Mode: plain NCM — computing class means from training set …")

    model.set_means_from_loader(train_loader, backbone=backbone, device=device)

    val_loss, val_acc = evaluate(model, val_loader, backbone, device)
    log.info(f"Val  loss={val_loss:.4f}  acc={val_acc*100:.2f}%")

    save_checkpoint(
        {"model": model.state_dict(), "epoch": 0, "val_acc": val_acc},
        ckpt_dir / "ncm_final.pt",
    )


# ---------------------------------------------------------------------------
# Mode: NCM-ML / NCM-MM (gradient-based W learning)
# ---------------------------------------------------------------------------

def run_ncm_ml(
    model: NCMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    backbone: nn.Module | None,
    device: torch.device,
    cfg: Dict,
    ckpt_dir: Path,
    start_epoch: int = 0,
) -> None:
    """
    SGD on the probabilistic NCM loss (paper Eq. 8) to learn W.
    Class means are updated as a running average after each epoch
    (or mini-batch, controlled by cfg.training.mean_update).

    Paper Sec. 3.2 / Alg. 1:
      - SGD with momentum + learning-rate decay (step schedule)
      - Means re-estimated once per epoch from current W-projected features
    """
    t_cfg  = cfg["training"]
    epochs = t_cfg["epochs"]
    lr     = t_cfg["lr"]
    wd     = t_cfg.get("weight_decay", 0.0)
    momentum = t_cfg.get("momentum", 0.9)
    lr_decay_epochs = t_cfg.get("lr_decay_epochs", [])
    lr_decay_factor = t_cfg.get("lr_decay_factor", 0.1)
    mean_update     = t_cfg.get("mean_update", "epoch")  # "epoch" | "batch"
    use_em          = cfg["model"].get("num_sub_means", 1) > 1

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=wd,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_decay_epochs, gamma=lr_decay_factor,
    )

    # ---- initialise means with one pass using identity W -----------------
    log.info("Initialising class means (one pass) …")
    model.set_means_from_loader(train_loader, backbone=backbone, device=device)

    best_acc   = 0.0
    best_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        epoch_loss = epoch_correct = epoch_total = 0

        for batch in train_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)

            if backbone is not None:
                with torch.no_grad():
                    feats = backbone(imgs)
            else:
                feats = imgs

            optimizer.zero_grad()
            loss = model.loss(feats, labels)
            loss.backward()
            optimizer.step()

            # Optional per-batch mean update (detached)
            if mean_update == "batch":
                with torch.no_grad():
                    model.update_means_batch(feats.detach(), labels)

            preds = model(feats).argmax(dim=-1)
            epoch_loss    += loss.item() * labels.size(0)
            epoch_correct += (preds == labels).sum().item()
            epoch_total   += labels.size(0)

        # Per-epoch mean re-estimation (default)
        if mean_update == "epoch":
            log.info("  Re-estimating class means …")
            model.set_means_from_loader(train_loader, backbone=backbone, device=device)

        # EM sub-mean update (NCM-MM)
        if use_em:
            log.info("  Running EM for sub-means …")
            _em_full_pass(model, train_loader, backbone, device,
                          n_iter=cfg["model"].get("em_iters", 5))

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, backbone, device)

        log.info(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  "
            f"time={time.time()-t0:.1f}s"
        )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_acc": val_acc,
        }
        save_checkpoint(state, ckpt_dir / "ncm_last.pt")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch + 1
            save_checkpoint(state, ckpt_dir / "ncm_best.pt")

    log.info(f"Best val_acc={best_acc*100:.2f}% at epoch {best_epoch}")


def _em_full_pass(
    model: NCMClassifier,
    loader: DataLoader,
    backbone: nn.Module | None,
    device: torch.device,
    n_iter: int,
) -> None:
    """One pass over the dataset for EM sub-mean update."""
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        if backbone is not None:
            with torch.no_grad():
                feats = backbone(imgs)
        else:
            feats = imgs
        model.em_update_sub_means(feats, labels, n_iter=n_iter)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NCM Training Script")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values, e.g. training.lr=0.01",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume",    type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)

    # ---- seed -------------------------------------------------------------
    set_seed(cfg.get("seed", 42))

    # ---- device -----------------------------------------------------------
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    log.info(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    train_loader, val_loader, num_classes = build_loaders(cfg)
    # Allow config to override auto-detected num_classes
    num_classes = cfg["model"].get("num_classes", num_classes)
    log.info(f"Dataset: {cfg['data'].get('dataset','imagefolder')} | "
             f"num_classes={num_classes}")

    # ---- backbone ---------------------------------------------------------
    backbone, feat_dim = build_backbone(cfg, device)
    if backbone is not None:
        log.info(f"Backbone: {cfg['backbone']['name']} (frozen)  feat_dim={feat_dim}")
    else:
        log.info(f"No backbone — raw features  feat_dim={feat_dim}")

    # ---- NCM model --------------------------------------------------------
    m_cfg = cfg["model"]
    model = NCMClassifier(
        feat_dim       = feat_dim,
        num_classes    = num_classes,
        metric_dim     = m_cfg.get("metric_dim", 0),
        num_sub_means  = m_cfg.get("num_sub_means", 1),
        temperature    = m_cfg.get("temperature", 1.0),
    ).to(device)
    log.info(f"NCMClassifier: {model}")

    # ---- checkpoint dir ---------------------------------------------------
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))

    # ---- optional resume --------------------------------------------------
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model)

    # ---- eval-only --------------------------------------------------------
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume <checkpoint>")
        val_loss, val_acc = evaluate(model, val_loader, backbone, device)
        log.info(f"[EVAL ONLY] val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")
        return

    # ---- training mode ----------------------------------------------------
    mode = cfg["training"].get("mode", "ncm_ml")
    log.info(f"Training mode: {mode}")

    if mode == "ncm":
        run_ncm(model, train_loader, val_loader, backbone, device, cfg, ckpt_dir)
    elif mode in ("ncm_ml", "ncm_mm"):
        run_ncm_ml(model, train_loader, val_loader, backbone, device,
                   cfg, ckpt_dir, start_epoch=start_epoch)
    else:
        raise ValueError(f"Unknown training mode: {mode!r}")


if __name__ == "__main__":
    main()
