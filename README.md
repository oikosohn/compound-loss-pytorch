# compound-loss-pytorch
Compound loss for pytorch


## [jaccard_ce_loss_smp.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/jaccard_ce_loss_smp.py)
Jaccard CE Loss from [Feature Pyramid Network for Multi-Class Land Segmentation](https://arxiv.org/abs/1806.03510) using segmentation_models.pytorch


## [unified_focal_loss_pytorch.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/unified_focal_loss_pytorch.py)
PyTorch implementation of [Unified Focal loss](https://arxiv.org/abs/2102.04525), ported from the
[official tensorflow repository](https://github.com/mlyg/unified-focal-loss).

Losses
- `SymmetricFocalLoss` / `AsymmetricFocalLoss`
- `SymmetricFocalTverskyLoss` / `AsymmetricFocalTverskyLoss`
- `SymmetricUnifiedFocalLoss` / `AsymmetricUnifiedFocalLoss`

### Usage
```python
from unified_focal_loss_pytorch import AsymmetricUnifiedFocalLoss

criterion = AsymmetricUnifiedFocalLoss(weight=0.5, delta=0.6, gamma=0.5)

# y_pred: softmax probabilities, y_true: one-hot labels
# (N, C, H, W) for 2D or (N, C, D, H, W) for 3D, channel 0 background / 1 foreground
loss = criterion(torch.softmax(logits, dim=1), y_true)
```

### Input contract
`y_pred` is **softmax probabilities**, `y_true` is **one-hot labels**, both channel-first
with exactly 2 channels: `(N, 2, H, W)` or `(N, 2, D, H, W)`, channel 0 background and
channel 1 foreground. Integer and boolean labels are cast automatically.

These are the mistakes that used to return a plausible but wrong number, and now raise:

| Input | Result |
| --- | --- |
| `(N, H, W, C)` channel-last, as in the tensorflow original | `ValueError` |
| `C != 2`, i.e. multi-class | `ValueError` - these losses are binary only, as in the original |
| `y_pred` and `y_true` shapes differ | `ValueError` - previously broadcast silently |
| Class-index labels `(N, H, W)` instead of one-hot | `ValueError` |

Two things are still on the caller, because neither can be checked without a pass over
the data on every step:

- **Logits instead of probabilities.** They are clamped to `[epsilon, 1-epsilon]` and
  produce a finite, meaningless loss. Apply `torch.softmax(logits, dim=1)` first.
- **`gamma >= 1` on the Focal Tversky losses** makes the loss diverge as the Dice score
  approaches 1. The original behaves the same way; the useful range is `0 <= gamma < 1`.

### Differences from the tensorflow original
- Tensors are channel-first, which fixes
  [[Issue #3] Incompatible pytorch tensor shape](https://github.com/oikosohn/compound-loss-pytorch/issues/3).
  The focal losses now also accept 3D volumes, which the original's hardcoded slicing does not.
- In the Focal Tversky losses, `(1-d) * (1-d)**-gamma` is folded into the equivalent
  `(1-d)**(1-gamma)`, which returns 0 instead of NaN when the Dice score reaches exactly 1.
- `AsymmetricUnifiedFocalLoss` now defaults to `gamma=0.5`, matching the original (it was `0.2`).
- The input validation above; the original has none.

All six losses are checked numerically against the original tensorflow implementation
across shapes, a `delta`/`gamma` grid, degenerate labels, and saturated predictions, plus
`torch.autograd.gradcheck` on each.

## References
```bibtex
@misc{https://doi.org/10.48550/arxiv.1806.03510,
  doi = {10.48550/ARXIV.1806.03510},
  url = {https://arxiv.org/abs/1806.03510},
  author = {Seferbekov, Selim S. and Iglovikov, Vladimir I. and Buslaev, Alexander V. and Shvets, Alexey A.},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Feature Pyramid Network for Multi-Class Land Segmentation},
  publisher = {arXiv},
  year = {2018},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{https://doi.org/10.48550/arxiv.2102.04525,
  doi = {10.48550/ARXIV.2102.04525},
  url = {https://arxiv.org/abs/2102.04525},
  author = {Yeung, Michael and Sala, Evis and Schönlieb, Carola-Bibiane and Rundo, Leonardo},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.6; J.3},
  title = {Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
