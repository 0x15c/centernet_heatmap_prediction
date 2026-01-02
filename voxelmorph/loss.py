import torch


def similarity_loss(fixed, warped):
    return torch.mean((fixed - warped) ** 2)


def smoothness_loss(flow):
    dx = torch.mean((flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2)
    dy = torch.mean((flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2)
    return dx + dy


def total_loss(fixed, warped, flow, smoothness_weight=0.1):
    sim = similarity_loss(fixed, warped)
    smooth = smoothness_loss(flow)
    return sim + smoothness_weight * smooth
