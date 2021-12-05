import segmentation_models_pytorch as smp
import torch.nn as nn

class JaccardCELoss(smp.utils.base.Loss):
    def __init__(self, alpha=1.0, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

        self.jaccardloss=smp.losses.JaccardLoss(mode='multiclass')
        self.jaccardloss.__name__ = 'jaccard_loss'

        self.celoss = nn.CrossEntropyLoss()
        self.celoss.__name__ = 'ce_loss'

    def forward(self, y_pred, y_true):
        return self.alpha * self.celoss.forward(y_pred, y_true) - self.beta * self.jaccardloss.forward(y_pred, y_true)