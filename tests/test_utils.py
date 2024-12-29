"""Tests for the `utils` module."""

import pytest
import torch

from torch_angular_search.utils import euler_ranges_from_symmetry_group


@pytest.mark.parametrize(
    "symmetry_group, symmetry_order, expected",
    [
        ("C", 1, [[-180.0, 180.0], [0, 180.0], [-180.0, 180.0]]),
        ("C", 2, [[-90.0, 90.0], [0, 180.0], [-180.0, 180.0]]),
        ("C", 3, [[-60.0, 60.0], [0, 180.0], [-180.0, 180.0]]),
        ("D", 1, [[-180.0, 180.0], [0, 90.0], [-180.0, 180.0]]),
        ("D", 4, [[-45.0, 45.0], [0, 90.0], [-180.0, 180.0]]),
        ("T", 1, [[-90.0, 90.0], [0, 54.7356], [-180.0, 180.0]]),
        ("O", 1, [[-45.0, 45.0], [0, 54.7356], [-180.0, 180.0]]),
        ("I", 1, [[-90.0, 90.0], [0, 31.7], [-180.0, 180.0]]),
    ],
)
def test_euler_ranges_from_symmetry_group(symmetry_group, symmetry_order, expected):
    """Test the euler_ranges_from_symmetry_group function."""
    result = euler_ranges_from_symmetry_group(symmetry_group, symmetry_order)

    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float64))


def test_euler_ranges_from_symmetry_group_error():
    """Ensure invalid symmetry group raises an error."""
    with pytest.raises(ValueError):
        euler_ranges_from_symmetry_group("Z", 1)
