"""PyTorch port of the Unified Focal loss (https://arxiv.org/abs/2102.04525).

Ported from the official tensorflow implementation:
https://github.com/mlyg/unified-focal-loss

Tensors are channel-first - (N, C, H, W) for 2D and (N, C, D, H, W) for 3D - instead of
the channel-last layout of the original. `y_pred` holds softmax probabilities and `y_true`
one-hot labels, with channel 0 as background and channel 1 as foreground.
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


# Channel-first is the one real difference from the tensorflow original, which makes a
# still-channel-last tensor the easiest mistake to make and the hardest to notice: every
# loss below happily returns a finite, plausible, wrong number for it. Same for a
# multi-class tensor, whose classes beyond the first two are silently ignored.

def check_input(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError('Shape mismatch: y_pred is {}, y_true is {}.'
                         .format(tuple(y_pred.shape), tuple(y_true.shape)))

    identify_axis(y_pred.shape)

    if y_pred.shape[1] != 2:
        raise ValueError('Expected 2 channels (background, foreground) at dim 1, got {}. '
                         'Tensors are channel-first: (N, C, H, W) or (N, C, D, H, W).'
                         .format(y_pred.shape[1]))


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
        check_input(y_pred, y_true)
        y_true = y_true.to(y_pred.dtype)

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
        check_input(y_pred, y_true)
        y_true = y_true.to(y_pred.dtype)

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
        check_input(y_pred, y_true)
        y_true = y_true.to(y_pred.dtype)

        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes.
        # The original writes this as (1-d) * (1-d)**-gamma, which is 0 * inf -> NaN once
        # a Dice score reaches exactly 1; the folded form is equal everywhere else.
        back_dice = torch.pow(1-dice_class[:,0], 1-self.gamma)
        fore_dice = torch.pow(1-dice_class[:,1], 1-self.gamma)

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
        check_input(y_pred, y_true)
        y_true = y_true.to(y_pred.dtype)

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
        fore_dice = torch.pow(1-dice_class[:,1], 1-self.gamma)

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
