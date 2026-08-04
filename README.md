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
channel 1 foreground.

Like the original, none of this is validated. These inputs return a finite, plausible,
wrong number rather than raising:

- **Channel-last `(N, H, W, C)`** - the tensorflow habit, and
  [issue #3](https://github.com/oikosohn/compound-loss-pytorch/issues/3) itself.
- **`C != 2`** - these losses are binary only; channels past the first two are never scored.
- **Mismatched `y_pred` / `y_true` shapes** - broadcasting quietly accepts several of them.
- **Logits instead of probabilities** - clamped to `[epsilon, 1-epsilon]`. Apply
  `torch.softmax(logits, dim=1)` first.
- **`uint8` masks** - `-y_true` wraps 1 to 255, so anything with a cross-entropy term is
  wrong. The Tversky losses never negate `y_true` and are unaffected. Cast masks to float
  or a signed integer dtype; `bool` raises rather than failing quietly.

And two numerical edges, both matching the original:

- **`gamma >= 1` on the Focal Tversky losses** diverges as the Dice score approaches 1.
  The useful range is `0 <= gamma < 1`.
- **`epsilon=0`** makes `(1-d) * (1-d)**-gamma` evaluate `0 * inf` at a perfect Dice score,
  giving NaN.

### Relationship to the tensorflow original
The only intentional change is the tensor layout. PyTorch convolutions emit `(N, C, H, W)`
where Keras emits `(N, H, W, C)`, so the spatial axes move from `[1,2]` to `[2,3]` and the
class slice from `y_pred[:,:,:,0]` to `y_pred[:,0]`. Both express the same thing; keeping
the original's literal indices is what produced issue #3. As a side effect the focal losses
work on 3D volumes, which the original's hardcoded slicing does not.

`AsymmetricUnifiedFocalLoss` had `gamma=0.2` in this file from its first commit. The
original and the paper both use `0.5`, so it now matches them - a change from the previous
version of this file, not from upstream, and it does change the default behaviour.

Everything else is a transcription. All six losses are checked numerically against the
original across shapes, a `delta`/`gamma` grid, degenerate labels, saturated predictions
and the edges above - including where the original returns NaN, which the port must too -
plus `torch.autograd.gradcheck` on CPU and CUDA.

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
