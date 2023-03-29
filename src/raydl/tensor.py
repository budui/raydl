from typing import Iterable, Optional, Union

import torch
from torch import FloatTensor, Size, Tensor, lerp, zeros_like
from torch.linalg import norm


def grid_transpose(
    tensors: Union[torch.Tensor, Iterable[torch.Tensor]], original_nrow: Optional[int] = None
) -> torch.Tensor:
    """
    batch tensors transpose.
    :param tensors: Tensor[(ROW*COL)*D1*...], or Iterable of same size tensors.
    :param original_nrow: original ROW
    :return: Tensor[(COL*ROW)*D1*...]
    """
    assert torch.is_tensor(tensors) or isinstance(tensors, Iterable)
    if not torch.is_tensor(tensors) and isinstance(tensors, Iterable):
        seen_size = None
        grid = []
        for tensor in tensors:
            if seen_size is None:
                seen_size = tensor.size()
                original_nrow = original_nrow or len(tensor)
            elif tensor.size() != seen_size:
                raise ValueError("expect all tensor in images have the same size.")
            grid.append(tensor)
        tensors = torch.cat(grid)
    assert original_nrow is not None
    assert isinstance(tensors, torch.Tensor)
    cell_size = tensors.size()[1:]
    tensors = tensors.reshape(-1, original_nrow, *cell_size)
    tensors = torch.transpose(tensors, 0, 1)
    return torch.reshape(tensors, (-1, *cell_size))


def slerp(v0: FloatTensor, v1: FloatTensor, t: Union[float, FloatTensor], DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0 or many-dimensional tensor
        DOT_THRESHOLD: Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1

    - many-dimensional vectors
    - v0 or v1 with last dim all zeroes, or v0 ~colinear with v1
        - falls back to lerp()
        - conditional logic implemented with parallelism rather than Python loops
    - many-dimensional tensor for t
        - you can ask for batches of slerp outputs by making t more-dimensional than the vectors
        -   slerp(
                v0:   torch.Size([2,3]),
                v1:   torch.Size([2,3]),
                t:  torch.Size([4,1,1]),
            )
        - this makes it interface-compatible with lerp()

    see more example at https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
    """
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0, dim=-1)
    v1_norm: FloatTensor = norm(v1, dim=-1)

    v0_normed: Tensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: Tensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: Tensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: Tensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    out = zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        v0_l = v0.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(
            dim=-1, sizes=(-1, *[1] * (v0.dim() - 2), v0.size(-1))
        )
        v1_l = v1.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(
            dim=-1, sizes=(-1, *[1] * (v1.dim() - 2), v1.size(-1))
        )

        lerped = lerp(v0_l, v1_l, t)

        out = out.masked_scatter_(gotta_lerp.unsqueeze(-1).expand(*t_batch_dims, *gotta_lerp.shape, 1), lerped)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        v0_s = v0.masked_select(can_slerp.unsqueeze(-1)).unflatten(
            dim=-1, sizes=(-1, *[1] * (v0.dim() - 2), v0.size(-1))
        )
        v1_s = v1.masked_select(can_slerp.unsqueeze(-1)).unflatten(
            dim=-1, sizes=(-1, *[1] * (v1.dim() - 2), v1.size(-1))
        )
        dot_s = dot.masked_select(can_slerp).unflatten(dim=-1, sizes=(-1, *[1] * (v1.dim() - 1)))

        # Calculate initial angle between v0 and v1
        theta_0: Tensor = dot_s.arccos()
        sin_theta_0: Tensor = theta_0.sin()
        # Angle at timestep t
        theta_t: Tensor = theta_0 * t
        sin_theta_t: Tensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: Tensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: Tensor = sin_theta_t / sin_theta_0
        slerped: Tensor = s0 * v0_s + s1 * v1_s

        return out.masked_scatter_(can_slerp.unsqueeze(-1).expand(*t_batch_dims, *can_slerp.shape, 1), slerped)

    return out
