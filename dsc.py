import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.spatial.distance import cdist
from skimage.measure import find_contours
import numpy as np

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):

        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def DICESEN_loss(input, target):
    smooth = 0.00000001
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    intersection = torch.sum(torch.mul(y_true_f,y_pred_f))
    dice= (2. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + torch.mul(y_pred_f,y_pred_f).sum() + smooth)
    sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)
    return 2-dice-sen   

# Define a function to compute the NSD metric
def compute_nsd(pred_mask, gt_mask, tolerance=1.0):
    """
    Compute Normalized Surface Dice (NSD) between two binary masks.
    
    Parameters:
    pred_mask (numpy.ndarray): Binary predicted mask
    gt_mask (numpy.ndarray): Binary ground truth mask
    tolerance (float): The surface tolerance in pixels

    Returns:
    float: NSD score
    """
    # Find contours of predicted and ground truth masks
    pred_contours = find_contours(pred_mask, level=0.5)
    gt_contours = find_contours(gt_mask, level=0.5)

    # Convert contours to a list of points
    pred_points = np.vstack(pred_contours) if pred_contours else np.array([])
    gt_points = np.vstack(gt_contours) if gt_contours else np.array([])

    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0  # If no valid contours, NSD is zero

    # Compute distance matrices
    dist_pred_to_gt = cdist(pred_points, gt_points)
    dist_gt_to_pred = cdist(gt_points, pred_points)

    # Check if points are within the tolerance
    pred_within_tol = np.any(dist_pred_to_gt <= tolerance, axis=1).sum()
    gt_within_tol = np.any(dist_gt_to_pred <= tolerance, axis=1).sum()

    # Compute NSD
    nsd_score = (pred_within_tol + gt_within_tol) / (len(pred_points) + len(gt_points))

    return nsd_score


class DiceSensitivityLoss(nn.Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(DiceSensitivityLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth = 1.):

        if self.n_classes == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)

        y_true_f = inputs.view(-1)
        y_pred_f = targets.view(-1)

        intersection = (y_true_f * y_pred_f).sum()

        dice= (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)

        sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)

        return 2 - dice-sen