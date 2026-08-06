# compound-loss-pytorch
Compound loss for pytorch


## [jaccard_ce_loss_smp.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/jaccard_ce_loss_smp.py)
Jaccard CE Loss from [Feature Pyramid Network for Multi-Class Land Segmentation](https://arxiv.org/abs/1806.03510) using segmentation_models.pytorch


## [unified_focal_loss_pytorch.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/unified_focal_loss_pytorch.py)
PyTorch implementation of [Unified Focal loss](https://arxiv.org/abs/2102.04525), ported from the
[official tensorflow repository](https://github.com/mlyg/unified-focal-loss).

All twelve functions of the original's `loss_functions.py`, in the same order.

The Unified Focal loss - binary only, channel 0 background and channel 1 foreground
- `SymmetricFocalLoss` / `AsymmetricFocalLoss`
- `SymmetricFocalTverskyLoss` / `AsymmetricFocalTverskyLoss`
- `SymmetricUnifiedFocalLoss` / `AsymmetricUnifiedFocalLoss`

The baselines the paper compares against
- `DiceLoss` / `TverskyLoss` - the same formula, `delta` 0.5 against 0.7
- `FocalTverskyLoss` / `FocalLoss` / `ComboLoss`
- `DiceCoefficient` - a metric, not a loss: it rises as the prediction improves

### Usage
```python
from unified_focal_loss_pytorch import AsymmetricUnifiedFocalLoss

criterion = AsymmetricUnifiedFocalLoss(weight=0.5, delta=0.6, gamma=0.5)

# y_pred: softmax probabilities, y_true: one-hot labels
# (N, C, H, W) for 2D or (N, C, D, H, W) for 3D, channel 0 background / 1 foreground
loss = criterion(torch.softmax(logits, dim=1), y_true)
```

`AsymmetricUnifiedFocalLoss` defaulted to `gamma=0.2` in this file until now. The original
and the paper both use `0.5`, so it now matches them - this changes existing behaviour.

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
