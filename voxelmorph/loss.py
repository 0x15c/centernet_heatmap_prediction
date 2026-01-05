import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


def ncc_2d(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    window_size: int = 9,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Simplified 2D Local Normalized Cross-Correlation for [1, 1, H, W] tensors.

    Returns a scalar tensor representing the mean NCC across the spatial dimensions.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Shapes must match. Got {tensor1.shape} and {tensor2.shape}")

    # Ensure window_size is a list for 2D
    win = [window_size, window_size]

    # Create a sum filter (1, 1, win, win)
    # This acts as a box-blur to calculate local sums
    sum_filt = torch.ones(
        1, 1, *win, device=tensor1.device, dtype=tensor1.dtype)

    padding = window_size // 2
    win_size = window_size * window_size

    # Compute element-wise products
    I2 = tensor1 * tensor1
    J2 = tensor2 * tensor2
    IJ = tensor1 * tensor2

    # Local sums using 2D convolution
    # Input: [1, 1, H, W], Filter: [1, 1, K, K] -> Output: [1, 1, H, W]
    I_sum = F.conv2d(tensor1, sum_filt, padding=padding)
    J_sum = F.conv2d(tensor2, sum_filt, padding=padding)
    I2_sum = F.conv2d(I2, sum_filt, padding=padding)
    J2_sum = F.conv2d(J2, sum_filt, padding=padding)
    IJ_sum = F.conv2d(IJ, sum_filt, padding=padding)

    # Local means
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    # Local Cross-covariance: cov(I, J) * win_size
    # Mathematically: sum((I - uI)(J - uJ)) = sum(IJ) - uI*sum(J) - uJ*sum(I) + uI*uJ*win_size
    cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * win_size

    # Local Variances: var(I) * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    # Local Squared correlation coefficient
    # We add eps to the denominator to prevent division by zero in uniform areas
    cc = (cross * cross) / (I_var * J_var + eps)

    # Return the mean across the spatial dimensions (H, W)
    # cc is [1, 1, H, W], so mean(dim=(2,3)) returns [1, 1]
    return cc.mean()


def similarity_loss(fixed, warped, loss_type=["MSE", "NCC"]):
    match loss_type:
        case "MSE":  # mean squared error
            return torch.mean((fixed - warped) ** 2)
        case "NCC":  # normalised cross correlation
            return ncc_2d(fixed, warped)


def smoothness_loss(flow):  # flow: [N, 2, H, W]
    # punish on smoothness, by the gradient norm.
    # let the displacement field be f = (u,v) where f: R^2 -> R^2
    # our goal is to obtain ||∇f||^2, 2-norm. ∇f is a 2 by 2 matrix,
    # we can calculate its squared norm by adding squared norm of column vectors of it
    # we first calculate the column vector of ∇f, which is (∂u/∂x, ∂u/∂y) and (∂v/∂x, ∂v/∂y)
    # then square them and add together
    ux2_plus_vx2 = torch.mean(
        (flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2)  # [N, 2, H, W-1]
    uy2_plus_vy2 = torch.mean(
        (flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2)  # [N, 2, H-1, W]
    return (ux2_plus_vx2 + uy2_plus_vy2)


def total_loss(fixed, warped, flow, smoothness_weight=0.1, sim_measure="MSE"):
    sim = similarity_loss(fixed, warped, loss_type=sim_measure)
    smooth = smoothness_loss(flow)
    return sim + smoothness_weight * smooth
