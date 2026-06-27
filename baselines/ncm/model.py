"""
NCM Classifier — Mensink et al., "Distance-Based Image Classification:
Generalizing to New Classes at Near-Zero Cost," IEEE TPAMI 2013.

Implements:
  1. Vanilla NCM          — Euclidean distance to class means (Sec. 3.1)
  2. NCM-ML               — Mahalanobis metric W learned via SGD (Sec. 3.2)
  3. NCM-Multiclass (MM)  — per-class sub-means via EM (Sec. 4)

The backbone (if any) is kept *frozen*; only W and/or the class means
are updated.  All mutable state lives in plain torch Tensors so the
whole object is trivially checkpoint-able via state_dict().
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone wrapper (frozen)
# ---------------------------------------------------------------------------

class FrozenBackbone(nn.Module):
    """Wraps any torchvision / timm model and freezes all parameters."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Core NCM
# ---------------------------------------------------------------------------

class NCMClassifier(nn.Module):
    """
    Nearest Class Mean classifier with optional low-rank metric learning.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of the (backbone) feature space.
    num_classes : int
        Number of classes seen so far (can be extended online).
    metric_dim : int
        Rank of the projection W ∈ R^{metric_dim × feat_dim}.
        Set to 0 to use plain Euclidean distance (no W).
    num_sub_means : int
        Number of sub-means per class for the NCM-MM variant.
        Set to 1 for plain NCM / NCM-ML.
    temperature : float
        Softmax temperature τ used when computing the probabilistic loss
        (Eq. 8 in the paper).  Only affects training.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        metric_dim: int = 0,
        num_sub_means: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.metric_dim = metric_dim
        self.num_sub_means = num_sub_means
        self.temperature = temperature

        # ---- learnable metric W (identity init → Euclidean start) ----------
        # Paper Sec 3.2: d²(x, μ_y) = ‖W(x − μ_y)‖²
        if metric_dim > 0:
            W_init = torch.eye(metric_dim, feat_dim)
            self.W = nn.Parameter(W_init)          # (metric_dim, feat_dim)
        else:
            self.W = None

        # ---- class means / sub-means (non-parametric, updated via SGD or
        #       running average)  -----------------------------------------
        # Shape: (num_classes, num_sub_means, feat_dim)
        self.register_buffer(
            "means",
            torch.zeros(num_classes, num_sub_means, feat_dim),
        )
        # Per-(class, sub-mean) sample counts for online mean update
        self.register_buffer(
            "counts",
            torch.zeros(num_classes, num_sub_means, dtype=torch.long),
        )
        # Sub-mean mixture weights π_{y,k}  (Sec. 4, Eq. 10)
        self.register_buffer(
            "mix_weights",
            torch.ones(num_classes, num_sub_means) / num_sub_means,
        )

    # ------------------------------------------------------------------
    # Projection helper
    # ------------------------------------------------------------------

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply W if metric learning is enabled, else pass through."""
        if self.W is not None:
            return x @ self.W.t()           # (N, metric_dim)
        return x                             # (N, feat_dim)

    def _project_means(self) -> torch.Tensor:
        """Return projected means: (C, K, d') where d' = metric_dim or feat_dim."""
        mu = self.means                      # (C, K, feat_dim)
        if self.W is not None:
            # broadcast matmul over class / sub-mean dimensions
            return mu @ self.W.t()           # (C, K, metric_dim)
        return mu

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _sq_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute squared distances from projected features z to all class
        (sub-)means in projected space.

        Returns
        -------
        dists : Tensor (N, C)
            Minimum squared distance over sub-means for each class.
        """
        mu_proj = self._project_means()          # (C, K, d')
        z_exp = z.unsqueeze(1).unsqueeze(1)      # (N, 1, 1, d')
        # (N, C, K)
        sq_dists_all = ((z_exp - mu_proj) ** 2).sum(-1)

        if self.num_sub_means == 1:
            return sq_dists_all.squeeze(-1)      # (N, C)

        # NCM-MM: soft-min via mixture weights (Eq. 10 in paper)
        # d²(x, y) = Σ_k π_{y,k} · d²(x, μ_{y,k})
        w = self.mix_weights.unsqueeze(0)        # (1, C, K)
        return (w * sq_dists_all).sum(-1)        # (N, C)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, feat_dim)  — raw (backbone) features.

        Returns
        -------
        logits : Tensor (N, C)
            Negative squared distances (acts as logits for softmax).
            Paper Eq. 8: p(y|x) ∝ exp(−d²(Wx, Wμ_y) / τ)
        """
        z = self._project(x)                     # (N, d')
        sq_dists = self._sq_distances(z)         # (N, C)
        return -sq_dists / self.temperature      # higher = closer = better

    # ------------------------------------------------------------------
    # Loss  (paper Eq. 8 — multiclass cross-entropy on the NCM logits)
    # ------------------------------------------------------------------

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.cross_entropy(logits, y)

    # ------------------------------------------------------------------
    # Mean-update utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_means_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Online running-mean update (Welford-style) for each sample.
        Used for plain NCM (no metric learning) or to initialise means.

        Only updates sub-mean index 0; for NCM-MM use the EM update.
        """
        for xi, yi in zip(x, y):
            c = yi.item()
            self.counts[c, 0] += 1
            delta = xi - self.means[c, 0]
            self.means[c, 0] += delta / self.counts[c, 0].float()

    @torch.no_grad()
    def set_means_from_loader(
        self,
        loader,
        backbone: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        """
        Compute exact class means from a DataLoader (one full pass).
        Optionally passes data through a frozen backbone first.
        """
        self.means.zero_()
        self.counts.zero_()

        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            if backbone is not None:
                with torch.no_grad():
                    feats = backbone(imgs)
            else:
                feats = imgs

            for feat, lbl in zip(feats, labels):
                c = lbl.item()
                self.counts[c, 0] += 1
                self.means[c, 0] += feat

        # Normalise
        cnt = self.counts[:, 0].float().clamp(min=1).unsqueeze(-1)
        self.means[:, 0] /= cnt

    # ------------------------------------------------------------------
    # EM update for NCM-MM sub-means (Sec. 4, Eqs. 11-14)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def em_update_sub_means(
        self, x: torch.Tensor, y: torch.Tensor, n_iter: int = 5
    ) -> None:
        """
        Run n_iter E-M steps to update sub-means and mixing weights
        for the classes present in the batch.

        Paper Sec. 4: Eq. 11 (E-step responsibilities), Eq. 14 (M-step).
        """
        if self.num_sub_means == 1:
            return

        classes = y.unique()
        for c in classes:
            mask = y == c
            xc = x[mask]                        # (Nc, D)

            mu = self.means[c]                  # (K, D)
            pi = self.mix_weights[c]            # (K,)

            for _ in range(n_iter):
                # E-step: responsibilities  r_{n,k} ∝ π_k exp(−‖x_n−μ_k‖²)
                sq = ((xc.unsqueeze(1) - mu) ** 2).sum(-1)  # (Nc, K)
                log_r = torch.log(pi.clamp(1e-9)) - sq
                r = torch.softmax(log_r, dim=-1)             # (Nc, K)

                # M-step: update means and weights
                r_sum = r.sum(0).clamp(min=1e-9)             # (K,)
                mu = (r.t() @ xc) / r_sum.unsqueeze(-1)     # (K, D)
                pi = r_sum / r_sum.sum()

            self.means[c] = mu
            self.mix_weights[c] = pi

    # ------------------------------------------------------------------
    # Predict / evaluate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (N,)."""
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        preds = self.predict(x)
        return (preds == y).float().mean().item()

    # ------------------------------------------------------------------
    # state_dict helpers for clean checkpointing
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, num_classes={self.num_classes}, "
            f"metric_dim={self.metric_dim}, num_sub_means={self.num_sub_means}, "
            f"temperature={self.temperature}"
        )
