"""Orientation and rotation utility functions."""

import torch


def euler_ranges_from_symmetry_group(
    symmetry_group: str = "C",
    symmetry_order: int = 1,
) -> torch.Tensor:
    """
    Generate Euler angle (ZYZ convention) ranges based on symmetry group.

    Parameters
    ----------
    symmetry_group: str
        Symmetry group of the particle in [C, D, T, O, I]. Default is "C".
    symmetry_order:
        Order of the symmetry group. Default is 1.

    Returns
    -------
    torch.Tensor: Tensor of shape (3, 2)
        Minimum and maximum ranges for phi, theta, and psi angles in units of
        degrees.

    Examples
    --------
    >>> euler_ranges_from_symmetry_group("C", 2)
    tensor([[ -90.0000,   90.0000],
            [   0.0000,  180.0000],
            [-180.0000,  180.0000]], dtype=torch.float64)
    """
    # Convert to upper case
    symmetry_group = symmetry_group.upper()

    phi_max = 180.0
    theta_max = 180.0
    psi_max = 180.0

    # TODO: Convert to a switch statement or dictionary-based approach
    # TODO: Groups for higher orders
    if symmetry_group == "C":
        phi_max = 180 / float(symmetry_order)
    elif symmetry_group == "D":
        phi_max = 180 / float(symmetry_order)
        theta_max = 90.0
    elif symmetry_group == "T":
        phi_max = 90.0
        theta_max = 54.7356
    elif symmetry_group == "O":
        phi_max = 45.0
        theta_max = 54.7356
    elif symmetry_group == "I":
        phi_max = 90.0
        theta_max = 31.7
    else:
        raise ValueError("Symmetry group not recognized")

    phi_range = [-phi_max, phi_max]
    theta_range = [0, theta_max]
    psi_range = [-psi_max, psi_max]

    return torch.tensor([phi_range, theta_range, psi_range], dtype=torch.float64)
