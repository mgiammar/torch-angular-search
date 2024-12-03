"""Generate finer angular search around multiple selected Euler angles without missing sphere coverage."""

import torch
from torch_angular_search.hopf_angles import get_uniform_euler_angles

def refine_euler_angles_josh(
        best_angles: torch.Tensor, 
        coarse_in_plane_step: float = 2.5, 
        coarse_out_of_plane_step: float = 1.5, 
        fine_in_plane_step: float = 0.1, 
        fine_out_of_plane_step: float = 0.1,
        coarse_phi_range: torch.Tensor = torch.tensor([-180, 180], dtype=torch.float64),
        ) -> list:
    """
    Refine sampling around multiple selected Euler angles without missing sphere coverage.

    Args:
        best_angles (torch.Tensor): Tensor of shape (NumBest, 3) containing best [phi, theta, psi] angles in degrees.
        coarse_in_plane_step (float): Coarse step size for in-plane angles (phi, psi) in degrees.
        coarse_out_of_plane_step (float): Coarse step size for out-of-plane angle (theta) in degrees.
        fine_in_plane_step (float): Finer step size for in-plane angles (phi, psi) in degrees.
        fine_out_of_plane_step (float): Finer step size for out-of-plane angle (theta) in degrees.

    Returns:
        list: List of torch tensors of shape (N, 3) in degrees,
                      where N is the number of refined angles per best angle.
    """
    eps = 1e-10
    # Convert angles to radians
    coarse_in_plane_step_tensor = torch.deg2rad(torch.tensor(coarse_in_plane_step, dtype=torch.float64))
    coarse_out_of_plane_step_tensor = torch.deg2rad(torch.tensor(coarse_out_of_plane_step, dtype=torch.float64))
    fine_in_plane_step_tensor = torch.deg2rad(torch.tensor(fine_in_plane_step, dtype=torch.float64))
    fine_out_of_plane_step_tensor = torch.deg2rad(torch.tensor(fine_out_of_plane_step, dtype=torch.float64))
    best_angles = torch.deg2rad(best_angles)
    
    fine_theta_ranges = (best_angles[:, 1] - coarse_out_of_plane_step_tensor + fine_out_of_plane_step_tensor, best_angles[:, 1] + coarse_out_of_plane_step_tensor - fine_out_of_plane_step_tensor - eps)
    fine_psi_ranges = (best_angles[:, 2] - coarse_in_plane_step_tensor + fine_in_plane_step_tensor, best_angles[:, 2] + coarse_in_plane_step_tensor - fine_in_plane_step_tensor - eps)

    # Stack the tensors to create shape (n, 2)
    fine_theta_ranges_tensor = torch.rad2deg(torch.stack(fine_theta_ranges, dim=1))
    fine_psi_ranges_tensor = torch.rad2deg(torch.stack(fine_psi_ranges, dim=1))
    
    #now get the Hopf fibration for the phi angles using coarse.
    # i.e. for tall best thetas, get what the phi step would have been
    phi_max_step = coarse_phi_range[1] - coarse_phi_range[0]
    phi_step_all = torch.clamp(torch.abs(coarse_out_of_plane_step_tensor / torch.sin(best_angles[:, 1])), max=phi_max_step)
    phi_step_all = phi_max_step / torch.round(phi_max_step / phi_step_all)
    #get phi range
    fine_phi_ranges = (best_angles[:, 0] - phi_step_all + fine_out_of_plane_step_tensor, best_angles[:, 0] + phi_step_all - fine_out_of_plane_step_tensor - eps)
    fine_phi_ranges_tensor = torch.rad2deg(torch.stack(fine_phi_ranges, dim=1))


    #Now get angles using Hopf fibration
    euler_angles = get_uniform_euler_angles(
        in_plane_step=fine_in_plane_step,
        out_of_plane_step=fine_out_of_plane_step,
        phi_ranges=fine_phi_ranges_tensor,
        theta_ranges=fine_theta_ranges_tensor,
        psi_ranges=fine_psi_ranges_tensor
    )

    return euler_angles