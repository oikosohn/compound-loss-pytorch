"""PyTorch port of the Unified Focal loss (https://arxiv.org/abs/2102.04525).

Ported from the official tensorflow implementation:
https://github.com/mlyg/unified-focal-loss

All twelve functions of the original `loss_functions.py` are here, in the same order: the
six baselines the paper compares against, then the six that make up the Unified Focal loss.

Tensors are channel-first - (N, C, H, W) for 2D and (N, C, D, H, W) for 3D - instead of
the channel-last layout of the original. `y_pred` holds softmax probabilities and `y_true`
one-hot labels, with channel 0 as background and channel 1 as foreground. Losses are called
`criterion(y_pred, y_true)`, matching `torch.nn`, where the original takes `(y_true, y_pred)`.
"""

import torch
import torch.nn as nn


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]

    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#           Dice loss          #
################################
class DiceLoss(nn.Module):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def __init__(self, delta=0.5, smooth=0.000001):
        super().__init__()
        self.delta = delta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)

        # Calculate Dice score
        dice_class = (tp + self.smooth)/(tp + self.delta*fn + (1-self.delta)*fp + self.smooth)

        # Average class scores
        return torch.mean(1-dice_class)


################################
#         Tversky loss         #
################################
class TverskyLoss(nn.Module):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def __init__(self, delta=0.7, smooth=0.000001):
        super().__init__()
        self.delta = delta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        tversky_class = (tp + self.smooth)/(tp + self.delta*fn + (1-self.delta)*fp + self.smooth)

        # Average class scores
        return torch.mean(1-tversky_class)


################################
#       Dice coefficient       #
################################
class DiceCoefficient(nn.Module):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.

    This is a metric, not a loss - it rises as the prediction improves.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def __init__(self, delta=0.5, smooth=0.000001):
        super().__init__()
        self.delta = delta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + self.smooth)/(tp + self.delta*fn + (1-self.delta)*fp + self.smooth)

        # Average class scores
        return torch.mean(dice_class)


################################
#          Combo loss          #
################################
class ComboLoss(nn.Module):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.dice_coefficient = DiceCoefficient()

    def forward(self, y_pred, y_true):
        dice = self.dice_coefficient(y_pred, y_true)

        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if self.beta is not None:
            beta_weight = torch.tensor([self.beta, 1-self.beta], dtype=cross_entropy.dtype, device=cross_entropy.device)
            cross_entropy = beta_weight.view(-1, *([1] * (y_pred.dim() - 2))) * cross_entropy

        # sum over classes
        cross_entropy = torch.mean(torch.sum(cross_entropy, dim=1))

        if self.alpha is not None:
            return (self.alpha * cross_entropy) - ((1 - self.alpha) * dice)
        return cross_entropy - dice


################################
#      Focal Tversky loss      #
################################
class FocalTverskyLoss(nn.Module):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, smooth=0.000001, epsilon=1e-07):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        tversky_class = (tp + self.smooth)/(tp + self.delta*fn + (1-self.delta)*fp + self.smooth)

        # Average class scores
        return torch.mean(torch.pow(1-tversky_class, self.gamma))


################################
#          Focal loss          #
################################
class FocalLoss(nn.Module):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. alpha > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, alpha=None, gamma_f=2., epsilon=1e-07):
        super().__init__()
        self.alpha = alpha
        self.gamma_f = gamma_f
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if self.alpha is not None:
            alpha_weight = torch.as_tensor(self.alpha, dtype=cross_entropy.dtype, device=cross_entropy.device)
            # a per-class alpha weights the channel axis, which is 1 here and -1 in the original
            if alpha_weight.dim() : alpha_weight = alpha_weight.view(-1, *([1] * (y_pred.dim() - 2)))
            focal_loss = alpha_weight * torch.pow(1 - y_pred, self.gamma_f) * cross_entropy
        else:
            focal_loss = torch.pow(1 - y_pred, self.gamma_f) * cross_entropy

        # sum over classes
        return torch.mean(torch.sum(focal_loss, dim=1))


################################
#       Symmetric Focal loss   #
################################
class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,1], self.gamma) * cross_entropy[:,1]
        fore_ce = self.delta * fore_ce

        return torch.mean(back_ce + fore_ce)


################################
#     Asymmetric Focal loss    #
################################
class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = self.delta * cross_entropy[:,1]

        return torch.mean(back_ce + fore_ce)


#################################
# Symmetric Focal Tversky loss  #
#################################
class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma)

        # Average class scores
        return torch.mean(torch.stack([back_dice, fore_dice], dim=-1))


#################################
# Asymmetric Focal Tversky loss #
#################################
class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = 1-dice_class[:,0]
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma)

        # Average class scores
        return torch.mean(torch.stack([back_dice, fore_dice], dim=-1))


###########################################
#      Symmetric Unified Focal loss       #
###########################################
class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super().__init__()
        self.weight = weight
        self.symmetric_ftl = SymmetricFocalTverskyLoss(delta=delta, gamma=gamma)
        self.symmetric_fl = SymmetricFocalLoss(delta=delta, gamma=gamma)

    def forward(self, y_pred, y_true):
        # Obtain Symmetric Focal Tversky loss
        symmetric_ftl = self.symmetric_ftl(y_pred, y_true)

        # Obtain Symmetric Focal loss
        symmetric_fl = self.symmetric_fl(y_pred, y_true)

        # Return weighted sum of Symmetric Focal loss and Symmetric Focal Tversky loss
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)
        return symmetric_ftl + symmetric_fl


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super().__init__()
        self.weight = weight
        self.asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma)
        self.asymmetric_fl = AsymmetricFocalLoss(delta=delta, gamma=gamma)

    def forward(self, y_pred, y_true):
        # Obtain Asymmetric Focal Tversky loss
        asymmetric_ftl = self.asymmetric_ftl(y_pred, y_true)

        # Obtain Asymmetric Focal loss
        asymmetric_fl = self.asymmetric_fl(y_pred, y_true)

        # Return weighted sum of Asymmetric Focal loss and Asymmetric Focal Tversky loss
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)
        return asymmetric_ftl + asymmetric_fl
