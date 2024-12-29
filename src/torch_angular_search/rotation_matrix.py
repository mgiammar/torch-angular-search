"""Functions to convert between rotation matrices and euler angles."""

import einops
import torch


def multiply_rotation_matrices(
    base_matrices: torch.Tensor,  # Shape (n, 3, 3)
    rotation_matrices: torch.Tensor,  # Shape (m, 3, 3)
) -> torch.Tensor:
    """
    Multiply each base matrix by each rotation matrix.

    Args:
        base_matrices: Tensor of n rotation matrices, shape (n, 3, 3)
        rotation_matrices: Tensor of m rotation matrices, shape (m, 3, 3)

    Returns
    -------
        Tensor of shape (m, n, 3, 3) containing all combinations of rotated matrices
    """
    # Rearrange tensors for broadcasting using einops
    base_matrices = einops.rearrange(
        base_matrices, "n h w -> 1 n h w"
    )  # Shape: (1, n, 3, 3)
    rotation_matrices = einops.rearrange(
        rotation_matrices, "m h w -> m 1 h w"
    )  # Shape: (m, 1, 3, 3)

    # Broadcast and perform matrix multiplication
    # Result shape will be (m, n, 3, 3)
    result = torch.matmul(rotation_matrices, base_matrices)

    return result
