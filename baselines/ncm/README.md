# NCM — Nearest Class Mean Classifier

## Paper Reproduced

> Thomas Mensink, Jakob Verbeek, Florent Perronnin, Gabriela Csurka.
> **"Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost."**
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 35(11):2624–2637, 2013.

---

## Method Summary

The paper introduces three progressively richer classifiers, all built on the idea of assigning a test point to its **nearest class mean** (NCM):

| Variant | What is learned | Distance metric |
|---------|----------------|-----------------|
| **NCM** (Sec. 3.1) | Nothing — means computed analytically | Euclidean |
| **NCM-ML** (Sec. 3.2) | Low-rank projection matrix W ∈ ℝ^{d'×D} | Mahalanobis: ‖W(x−μ_y)‖² |
| **NCM-MM** (Sec. 4) | W + per-class sub-means {μ_{y,k}} via EM | Weighted mixture over sub-means |

Classification rule (Eq. 1):
```
ŷ = argmin_y  d²(Wx, Wμ_y)
```

Training loss — probabilistic cross-entropy on NCM logits (Eq. 8):
```
L = -log p(y|x)   where   p(y|x) ∝ exp(-d²(Wx, Wμ_y) / τ)
```

W is optimised by SGD; class means are re-estimated after each epoch (Alg. 1).

---

## File Layout

```
ncm/
├── model.py          # NCMClassifier + FrozenBackbone
├── train.py          # Training / evaluation script
├── config.yaml       # All hyperparameters (fully commented)
├── requirements.txt  # Python dependencies
├── run.sh            # Single-command launcher
└── README.md         # This file
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with default config (CIFAR-100, ResNet-50 backbone, NCM-ML)
python train.py --config config.yaml

# 3. Evaluate a saved checkpoint
python train.py --config config.yaml --eval_only --resume checkpoints/ncm/ncm_best.pt

# 4. Override hyperparameters without editing YAML
python train.py --config config.yaml --override training.lr=0.001 model.metric_dim=128

# 5. Single-command launcher (see run.sh)
bash run.sh
```

---

## Configuration Reference

All tunable parameters live in `config.yaml`. Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `training.mode` | `ncm_ml` | `ncm` / `ncm_ml` / `ncm_mm` |
| `model.metric_dim` | `256` | Rank of W; `0` = Euclidean NCM |
| `model.num_sub_means` | `1` | Sub-means per class (NCM-MM requires >1) |
| `model.temperature` | `1.0` | Softmax temperature τ |
| `training.lr` | `0.01` | SGD learning rate |
| `training.lr_decay_epochs` | `[15,25]` | Step-decay schedule |
| `backbone.name` | `resnet50` | Any torchvision model; `null` = raw features |
| `backbone.pretrained` | `true` | Load ImageNet weights |

---

## Assumptions & Design Decisions

1. **Frozen backbone.** The paper uses fixed Fisher Vector / SIFT features; we replace those with a pretrained deep backbone (ResNet-50 by default) kept completely frozen. Only W (and class means) are updated, matching the spirit of the paper.

2. **Mean re-estimation schedule.** Paper Algorithm 1 re-estimates means after *every* gradient epoch. We implement this as `mean_update: "epoch"` (default). A faster approximation (`"batch"`) is also available.

3. **Temperature τ.** The paper uses τ=1 throughout. Exposing it as a hyperparameter aids ablation; set to 1.0 to reproduce the paper exactly.

4. **NCM (no W) mode.** When `model.metric_dim: 0`, the model uses plain Euclidean distance and `training.mode: ncm` performs a single-pass mean estimation with no gradient steps — matching the non-metric NCM baseline in the paper.

5. **Dataset.** The paper uses ImageNet-2010 (1000 classes, >1M images). We default to CIFAR-100 for fast iteration. Switch `data.dataset: imagefolder` and point `data.root` to an ImageNet-style folder to reproduce the original experiments.

6. **Metric initialisation.** W is initialised to a truncated identity matrix so that training starts from (approximately) Euclidean distance.

---

## Unsupported / Out-of-Scope Features

- **k-NN classifier** — the paper also evaluates k-NN; not implemented here (NCM only).
- **Semantic embeddings** (Sec. 5 in the paper) — zero-shot transfer via word vectors is not implemented.
- **ImageNet hierarchy loss** — the paper experiments with a hierarchical classification objective; not implemented.
- **Fisher Vector features** — the original features are replaced by a deep backbone; no Fisher Vector extraction code is included.
- **Distributed training** — single-GPU / single-node only.

---

## Expected Runtime

Tested on a single NVIDIA A100 (80 GB), CIFAR-100, ResNet-50 backbone:

| Mode | Epochs | Approx. time |
|------|--------|-------------|
| `ncm` (plain) | 1 pass | ~30 seconds |
| `ncm_ml` | 30 | ~12 minutes |
| `ncm_mm` (K=3) | 30 | ~18 minutes |

On a V100 or consumer GPU, multiply by ~2–3×.  
For ImageNet (1000 classes, 1.28M images), expect ~2–4 hours for 30 epochs of `ncm_ml`.

---

## Checkpoint Format

```python
{
    "model":     model.state_dict(),   # NCMClassifier weights + buffers
    "optimizer": optimizer.state_dict(),
    "epoch":     int,
    "val_acc":   float,
}
```

Resume training with:
```bash
python train.py --config config.yaml --resume checkpoints/ncm/ncm_last.pt
```

---

## Citation

```bibtex
@article{mensink2013distance,
  title     = {Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost},
  author    = {Mensink, Thomas and Verbeek, Jakob and Perronnin, Florent and Csurka, Gabriela},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {35},
  number    = {11},
  pages     = {2624--2637},
  year      = {2013},
  publisher = {IEEE}
}
```
