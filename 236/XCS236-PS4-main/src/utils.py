from sampling_config import get_config
from typing import Dict, List, Optional
import torch
from torch import Tensor
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
from dataclasses import dataclass
from model import UNet
from diffusers import UNet2DModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Common configuration for inference methods."""
    num_steps: int
    seed: Optional[int] = None
    eta: float = 0.0  # Only used for DDIM

def get_alpha_bar_prev(
    scheduler_data: Dict[str, torch.Tensor], t_prev: int
) -> torch.Tensor:
    """Get alpha_bar value for previous timestep.

    Args:
        scheduler_data (Dict[str, torch.Tensor]): Scheduler parameters.
        t_prev (int): Previous timestep.

    Returns:
        torch.Tensor: The alpha_bar value for the previous timestep.
    """
    config = get_config()
    return (
        scheduler_data["alphas_bar"][t_prev]
        if t_prev >= 0
        else torch.tensor(1.0, device=config.device)
    )

def save_images(images: List[np.ndarray], prefix: str) -> None:
    """Save a list of images with the given prefix.

    Args:
        images (List[np.ndarray]): List of images to save.
        prefix (str): Prefix for the saved image filenames.

    Raises:
        Exception: If an error occurs during saving images.
    """
    try:
        config = get_config()
        save_dir = config.save_dir
        save_dir.mkdir(exist_ok=True, parents=True)

        for t, im in tqdm(images, desc="Saving images"):
            if im.shape[-1] == 1:
                im = im[:, :, 0]
            im = (im * config.image_scale).round().astype("uint8")
            pilim = Image.fromarray(im)
            save_path = save_dir / f"{prefix}_img_{t}.png"
            pilim.save(save_path)
            logger.debug(f"Saved image to {save_path}")
        logger.info(f"Image(s) saved to {save_dir} with prefix: {prefix}")
    except Exception as e:
        logger.error(f"Error saving images: {e}")
        raise

def read_and_format_image(path: str, device: str = "cpu") -> Tensor:
    """Read and format an image from the given path.

    Args:
        path (str): The path to the image file.
        device (str, optional): The device to load the image onto. Defaults to "cpu".

    Returns:
        Tensor: The formatted image tensor.
    """
    config = get_config()
    image = np.array(Image.open(path).convert("RGB"))
    image = (np.array(image) / 127.5) - 1  # Normalize the image to [-1, 1]
    image = (
        torch.tensor(image, dtype=torch.float32, device=config.device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    ).to(device)
    return image


def get_experiment_config(experiment_name: str = "ddim"):
    config = get_config()
    if experiment_name == "ddim":
        return InferenceConfig(num_steps=10, eta=0.0, seed=config.seed)
    elif experiment_name in ["ddpm", "inpaint"]:
        return InferenceConfig(num_steps=1000, seed=config.seed)
    else:
        raise ValueError("Unknown experiment type. Choose 'ddim', 'ddpm' or 'inpaint'.")


def model_factory(model_type: str, model_id: str, weights_path: str, device: str):
    """
    Factory function to create and return the appropriate model.

    Args:
        model_type (str): The type of model to create ('unet2d' or 'unet').
        model_id (str): The model ID for loading the UNet2DModel.
        weights_path (str): The path to the weights file for the UNet model.
        device (str): The device to load the model onto ('cpu', 'cuda', or 'mps').

    Returns:
        model: An instance of the specified model.
    """
    if model_type == "unet2d":
        # Load UNet2DModel from pretrained weights
        model = UNet2DModel.from_pretrained(model_id)
    elif model_type == "unet":
        # Initialize UNet and load local weights
        model = UNet(1)  # Adjust the parameters as needed
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        raise ValueError("Unknown model type. Choose 'unet2d' or 'unet'.")

    # Move the model to the specified device
    return model.to(device)