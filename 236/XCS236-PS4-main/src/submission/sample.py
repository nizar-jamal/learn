from typing import Dict, Tuple
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sampling_config import get_config
from utils import get_alpha_bar_prev

def get_timesteps(training_timesteps: int, num_steps: int) -> Tuple[Tensor, Tensor]:
    """
    Generate timesteps for the diffusion process.

    Args:
        training_timesteps (int): Total number of training timesteps.
        num_steps (int): Number of inference steps.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the timesteps and previous timesteps.
    """
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    # Downsample the timesteps by selecting every (training_timesteps // num_steps) index
    step_size = training_timesteps // num_steps
    timesteps: Tensor = torch.arange(0, training_timesteps, step_size, dtype=torch.long)[-num_steps:]
    timesteps = timesteps.flip(0)  # Reverse the order to start from the highest timestep
    prev_timesteps: Tensor = torch.cat([timesteps[1:], torch.tensor([-1], dtype=torch.long)])  # Lag by one with -1 padding
    return timesteps, prev_timesteps
    ### END CODE HERE ###

def predict_x0(
    predicted_noise: torch.Tensor,
    t: int,
    sample_t: torch.Tensor,
    scheduler_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict the original image from the noisy sample.

    Args:
        predicted_noise (torch.Tensor): The predicted noise tensor.
        t (int): Current timestep.
        sample_t (torch.Tensor): The noisy sample tensor.
        scheduler_params (Dict[str, torch.Tensor]): Scheduler parameters.

    Returns:
        torch.Tensor: The predicted original image tensor.
    """
    ### START CODE HERE ###
    alpha_bar_t: Tensor = scheduler_params["alphas_bar"][t]
    # Compute x0 using the denoising formula
    predicted_x0: Tensor = (sample_t - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
    
    # Clamp x0 to ensure pixel values are valid in the range [-1, 1]
    predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)
    
    return predicted_x0    ### END CODE HERE ###

def compute_forward_posterior_mean(
    predicted_x0: Tensor,
    noisy_image: Tensor,
    scheduler_params: Dict[str, Tensor],
    t: int,
    t_prev: int,
) -> Tensor:
    """Compute the mean of the forward posterior distribution.

    Args:
        predicted_x0 (Tensor): The predicted original image tensor.
        noisy_image (Tensor): The noisy image tensor.
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current timestep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed mean of the forward posterior distribution.
    """
    ### START CODE HERE ###
    # Extract scheduler parameters
    beta_t: Tensor = scheduler_params["betas"][t]
    alpha_t: Tensor = scheduler_params["alphas"][t]
    alpha_bar_t: Tensor = scheduler_params["alphas_bar"][t]
    alpha_bar_t_prev: Tensor = scheduler_params["alphas_bar"][t_prev]

    # Compute tilde_mu_t using the formula
    term1: Tensor = (torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)) * predicted_x0
    term2: Tensor = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * noisy_image
    tilde_mu_t: Tensor = term1 + term2

    return tilde_mu_t
    ### END CODE HERE ###

def compute_forward_posterior_variance(
    scheduler_params: Dict[str, Tensor], t: int, t_prev: int
) -> Tensor:
    """Compute the variance of the forward posterior distribution.

    Args:
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current tim estep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed variance of the forward posterior distribution.
    """
    ### START CODE HERE ###
    # Extract scheduler parameters
    beta_t: Tensor = scheduler_params["betas"][t]
    alpha_bar_t: Tensor = scheduler_params["alphas_bar"][t]
    alpha_bar_t_prev: Tensor = scheduler_params["alphas_bar"][t_prev]

    # Compute tilde_beta_t using the formula
    tilde_beta_t: Tensor = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t

    return tilde_beta_t
    ### END CODE HERE ###

def get_stochasticity_std(
    eta: float, t: int, t_prev: int, alphas_bar: torch.Tensor
) -> torch.Tensor:
    """Calculate the stochasticity standard deviation for DDIM sampling.

    Args:
        eta (float): The DDIM stochasticity parameter (0 = deterministic, 1 = full stochastic).
        t (int): Current timestep.
        t_prev (int): Previous timestep.
        alphas_bar (torch.Tensor): Cumulative product of (1 - beta).

    Returns:
        Tensor: The computed standard deviation.
    """
    ### START CODE HERE ###
    std = eta * ((1 - alphas_bar[t_prev]) / (1 - alphas_bar[t])).sqrt()
    return std
    ### END CODE HERE ###

def predict_sample_direction(
    alphas_bars_prev: float, predicted_noise: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Predict the direction for the next sample in DDIM.

    Args:
        alphas_bars_prev (float): Alpha bar value for previous timestep.
        predicted_noise (torch.Tensor): Predicted noise from the model.
        std (torch.Tensor): Standard deviation for the step.

    Returns:
        Tensor: The predicted sample direction.
    """
    ### START CODE HERE ###
    sample_direction: Tensor = (-std * predicted_noise) / torch.sqrt(torch.tensor(alphas_bars_prev))
    return sample_direction
    ### END CODE HERE ###


def stochasticity_term(std: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Compute the stochasticity term for DDIM sampling.

    Args:
        std (torch.Tensor): The computed standard deviation.
        noise (torch.Tensor): Random noise tensor.

    Returns:
        Tensor: The stochasticity term to be added to the sample.
    """
    ### START CODE HERE ###
    return std * noise
    ### END CODE HERE ###
