# compound-loss-pytorch
Compound loss for pytorch


### [jaccard_ce_loss_smp.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/jaccard_ce_loss_smp.py)
- PyTorch implementation of Jaccard CE Loss using segmentation_models.pytorch
    - Implement the loss of [Feature Pyramid Network for Multi-Class Land Segmentation](https://arxiv.org/abs/1806.03510)


### [unified_focal_loss_pytorch.py](https://github.com/oikosohn/compound-loss-pytorch/blob/main/unified_focal_loss_pytorch.py)
- PyTorch implementation of [Unified Focal loss](https://arxiv.org/abs/2102.04525)
    - I just converted the code in [Official tensorflow repository](https://github.com/mlyg/unified-focal-loss) into a PyTorch.
- Focal Tversky loss (symmetric and asymmetric), Focal loss (symmetric and asymmetric), Unified Focal loss (symmetric and asymmetric)
