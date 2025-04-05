import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
from torch import Tensor
from typing import Dict
from sampling_config import get_config

def add_forward_tnoise(
    image: Tensor, timestep: int, scheduler_data: Dict[str, Tensor]
) -> Tensor:
    """Add forward timestep noise to the image.

    Args:
        image (Tensor): The input image tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.

    Returns:
        x_t (Tensor): The image tensor with added noise.
    """
    config = get_config()
    alpha_bar_at_t = scheduler_data["alphas_bar"][timestep]
    noise = torch.randn(image.shape, device=config.device)
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###

def apply_inpainting_mask(
        original_image: Tensor, 
        noisy_image: Tensor,
        mask: Tensor, 
        timestep, 
        scheduler_data) -> Tensor:
    """Apply the inpainting mask to the image.

    Args:
        image (Tensor): The input image tensor.
        noisy_image (Tensor): The noisy image tensor.
        mask (Tensor): The inpainting mask tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
    Returns:
        Tensor: The inpainted image tensor.
    
    HINT: use add_forward_tnoise to add noise to the original image.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###

def get_mask(image: Tensor) -> Tensor:
    """Generate a mask for the given image.

    Args:
        image (Tensor): The input image tensor.

    Returns:
        Tensor: The generated mask tensor.
    """
    # Suppose your image is [1, 3, H, W]
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###