from typing import Dict

import torch

from .score_matching_utils import (
    add_noise,
    compute_divergence,
    compute_gaussian_score,
    compute_l2norm_squared,
    compute_score,
    compute_target_score,
    log_p_theta,
)


# Objective Function for Denoising Score Matching
def denoising_score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor], noise_std: float = 0.1
) -> torch.Tensor:
    """Objective function for denoising score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.
        noise_std (float): Standard deviation of the noise to add.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    # Add Gaussian noise to the input tensor
    noisy_x = add_noise(x, noise_std)

    # Compute the target score for denoising
    target_score = compute_target_score(x, noisy_x, noise_std)

    # Compute the model's score function for the noisy inputs
    model_score = compute_gaussian_score(noisy_x, mean, log_var)

    # Compute the squared L2 norm of the difference between target and model scores
    score_diff = target_score - model_score
    objective = compute_l2norm_squared(score_diff).mean()  # Average over batch

    return objective
    ### END CODE HERE ###


# Objective Function for Score Matching
def score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Objective function for score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]

    ### START CODE HERE ###
    # Compute log probability of x using the Gaussian distribution defined by theta
    log_p = log_p_theta(x, mean, log_var)

    # Compute the score function ∇_x log p(x)
    score = compute_score(log_p, x)

    # Compute divergence of the score function ∇_x ⋅ sθ(x)
    divergence = compute_divergence(score, x)

    # Compute the objective value as ||sθ(x)||^2 + 2 * divergence
    l2_norm_squared = compute_l2norm_squared(score)
    objective = l2_norm_squared.mean() + 2 * divergence.mean()

    return objective
    ### END CODE HERE ###
