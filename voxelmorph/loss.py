import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class NCC2D:
    """
    Local normalized cross-correlation loss for 2D images only.

    Input tensors are assumed to be shaped:
        [B, C, H, W]

    This is the PyTorch translation of the TensorFlow version,
    restricted to 2D only.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        # Ii, Ji: [B, C, H, W]
        assert Ii.ndim == 4 and Ji.ndim == 4, "Inputs must be 4D tensors [B, C, H, W]"
        assert Ii.shape == Ji.shape, "Ii and Ji must have the same shape"

        B, C, H, W = Ii.shape

        # set window size
        if self.win is None:
            win = [9, 9]
        elif isinstance(self.win, int):
            win = [self.win, self.win]
        else:
            assert len(self.win) == 2, "For 2D case, win must be int or length-2 list/tuple"
            win = list(self.win)

        kh, kw = win

        # compute CC terms
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # depthwise convolution filter: one all-ones kernel per channel
        # shape for grouped conv2d: [C, 1, kh, kw]
        sum_filt = torch.ones((C, 1, kh, kw), dtype=Ii.dtype, device=Ii.device)

        # SAME padding for odd kernel sizes
        pad_h = kh // 2
        pad_w = kw // 2

        # local sums via grouped conv
        I_sum = F.conv2d(Ii, sum_filt, stride=1, padding=(pad_h, pad_w), groups=C)
        J_sum = F.conv2d(Ji, sum_filt, stride=1, padding=(pad_h, pad_w), groups=C)
        I2_sum = F.conv2d(I2, sum_filt, stride=1, padding=(pad_h, pad_w), groups=C)
        J2_sum = F.conv2d(J2, sum_filt, stride=1, padding=(pad_h, pad_w), groups=C)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=1, padding=(pad_h, pad_w), groups=C)

        # compute local means
        win_size = kh * kw * C
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # cross / variances
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.clamp(cross, min=self.eps)

        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.clamp(I_var, min=self.eps)

        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.clamp(J_var, min=self.eps)

        if self.signed:
            cc = cross / torch.sqrt(I_var * J_var + self.eps)
        else:
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        cc = self.ncc(y_true, y_pred)

        if reduce == 'mean':
            cc = cc.flatten(start_dim=1).mean(dim=1)
        elif reduce == 'max':
            cc = cc.flatten(start_dim=1).max(dim=1).values
        elif reduce is not None:
            raise ValueError(f"Unknown NCC reduction type: {reduce}")

        return -cc


def similarity_loss(fixed, warped, loss_type=["MSE", "NCC"]):
    match loss_type:
        case "MSE":  # mean squared error
            return torch.mean((fixed - warped) ** 2)
        case "NCC":  # normalised cross correlation
            NCC_Loss = NCC2D(win=9, signed=False)
            return NCC_Loss(fixed, warped)


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
