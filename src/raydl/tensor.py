import torch


def slerp(x, y, t):
    """
    Spherical interpolation of a batch of vectors.
    """

    norm_x = x / x.norm(dim=-1, keepdim=True)
    norm_y = y / y.norm(dim=-1, keepdim=True)
    cos_xy = (norm_x * norm_y).sum(dim=-1, keepdim=True)

    theta = torch.acos(cos_xy)
    return (torch.sin((1 - t) * theta) * x + torch.sin(t * theta) * y) / torch.sin(theta)
