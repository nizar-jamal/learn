import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch import Tensor
from tqdm import tqdm

from sampling_config import set_config, get_config
from utils import InferenceConfig, get_alpha_bar_prev, save_images, read_and_format_image, get_experiment_config, model_factory
import importlib.util

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import (
    get_mask,
    apply_inpainting_mask,
    get_timesteps,
    predict_x0,
    compute_forward_posterior_mean,
    compute_forward_posterior_variance,
    get_stochasticity_std,
    predict_sample_direction,
    stochasticity_term
  )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run DDPM sampling.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "faces"],
        required=True,
        help="Type of dataset to run: 'mnist' or 'faces'.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["ddim", "ddpm", "inpaint"],
        required=True,
        help="Type of experiment to run: 'ddim', 'ddpm' or 'inpaint.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Image path for inpainting",
    )
    return parser.parse_args()


@dataclass
class SchedulerParams:
    """Parameters for the noise scheduler."""

    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_train_timesteps: int = 1000

    def build_inference_scheduler_parameters(self) -> Dict[str, Any]:
        """Build the scheduler parameters for inference.

        Returns:
            Dict[str, Any]: A dictionary containing the scheduler parameters.
        """
        betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps
        ).to(config.device)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        return {
            "steps": self.num_train_timesteps,
            "betas": betas,
            "alphas": alphas,
            "alphas_bar": alphas_bar,
        }


def ddim_inference(
    unet: UNet2DModel,
    scheduler_data: Dict[str, Tensor],
    num_steps: int,
    eta: float = 0.0,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Perform DDIM inference.

    Args:
        unet (UNet2DModel): The UNet model.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
        num_steps (int): Number of inference steps.
        eta (float, optional): DDIM stochasticity parameter (0 = deterministic, 1 = full stochastic). Defaults to 0.0.
        seed (Optional[int], optional): Optional random seed. Defaults to None.
        generator (Optional[torch.Generator], optional): Optional random number generator. Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: (final_image, intermediate_images).
    """
    try:
        config = get_config()
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=config.device).manual_seed(seed)
        elif generator is None:
            generator = torch.Generator(device=config.device)

        timesteps, timesteps_prev = get_timesteps(scheduler_data["steps"], num_steps)

        image_shape = (
            1,
            unet.in_channels,
            unet.sample_size,
            unet.sample_size,
        )
        # Create noise
        noisy_image = torch.randn(
            image_shape, generator=generator, device=config.device
        )

        images = []
        for timestep, timestep_prev in tqdm(
            list(zip(timesteps, timesteps_prev)), desc="DDIM Sampling"
        ):
            t = timestep.item()
            t_prev = timestep_prev.item()
            alpha_bars_prev = get_alpha_bar_prev(scheduler_data, t_prev)
            predicted_noise = unet(noisy_image, timestep).sample

            predicted_x0 = predict_x0(predicted_noise, t, noisy_image, scheduler_data)

            if eta == 0:
                sample_direction = (1 - alpha_bars_prev) ** 0.5 * predicted_noise
            elif eta > 0:
                std = get_stochasticity_std(
                    eta, t, t_prev, scheduler_data["alphas_bar"]
                )
                sample_direction = predict_sample_direction(
                    alpha_bars_prev, predicted_noise, std
                )
           
            noisy_image = (alpha_bars_prev**0.5) * predicted_x0 + sample_direction
            if eta > 0:
                stochasticity_signal = torch.randn(
                    image_shape, generator=generator, device=config.device
                )
                noisy_image += stochasticity_term(std, stochasticity_signal)
            images.append(
                [
                    t,
                    (noisy_image / 2 + 0.5)
                    .clamp(0, 1)
                    .cpu()
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .numpy(),
                ]
            )
        
        image = (noisy_image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().squeeze(0).permute((1, 2, 0)).numpy()
        return image, images
    except Exception as e:
        logger.error(f"Error during DDIM inference: {e}")
        raise


def ddpm_inference(
    unet: UNet2DModel,
    scheduler_data: Dict[str, Tensor],
    num_steps: int,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    original_image: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Perform DDPM inference.

    Args:
        unet (UNet2DModel): The UNet model.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
        num_steps (int): Number of inference steps.
        seed (Optional[int], optional): Optional random seed. Defaults to None.
        generator (Optional[torch.Generator], optional): Optional random number generator. Defaults to None.
        original_image (Optional[Tensor], optional): Original image for inpainting. Defaults to None.
        mask (Optional[Tensor], optional): Mask for inpainting. Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: (final_image, intermediate_images).
    """
    try:
        config = get_config()
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=config.device).manual_seed(seed)
        elif generator is None:
            generator = torch.Generator(device=config.device)

        timesteps, timesteps_prev = get_timesteps(scheduler_data["steps"], num_steps)

        image_shape = (1, unet.in_channels, unet.sample_size, unet.sample_size)

        noisy_image = torch.randn(
            image_shape, generator=generator, device=config.device
        )

        images = []
        for timestep, timestep_prev in tqdm(
            list(zip(timesteps, timesteps_prev)), desc="DDPM Sampling"
        ):
            t = timestep.item()
            t_prev = timestep_prev.item()

            if original_image is not None:
                if mask is not None:
                    noisy_image = apply_inpainting_mask(
                        original_image, noisy_image, mask, t, scheduler_data
                    )
                else:
                    raise ValueError(
                        "Missing the mask, which is needed for inpainting ..."
                    )
            predicted_noise = unet(noisy_image, timestep).sample
            predicted_x0 = predict_x0(predicted_noise, t, noisy_image, scheduler_data)

            forward_posterior_mean = compute_forward_posterior_mean(
                predicted_x0, noisy_image, scheduler_data, t, t_prev
            )
            noisy_image = forward_posterior_mean

            if t > 0:
                forward_posterior_variance = compute_forward_posterior_variance(
                    scheduler_data, t, t_prev
                )
                noise = torch.randn(
                    predicted_noise.shape, generator=generator, device=config.device
                ) * torch.sqrt(forward_posterior_variance)
                noisy_image += noise

            images.append(
                [
                    t,
                    (noisy_image / 2 + 0.5)
                    .clamp(0, 1)
                    .cpu()
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .numpy(),
                ]
            )
        return noisy_image, images
    except Exception as e:
        logger.error(f"Error during DDPM inference: {e}")
        raise


def run_inference(
    method: Literal["ddim", "ddpm", "inpaint"],
    unet: UNet2DModel,
    scheduler_data: Dict[str, Tensor],
    inference_config: InferenceConfig,
    generator: Optional[torch.Generator] = None,
    image_path: str = "",
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Unified interface for running inference.

    Args:
        method (Literal["ddim", "ddpm", "inpaint"]): The sampling method to use.
        unet (UNet2DModel): The UNet model.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
        inference_config (InferenceConfig): Inference configuration.
        generator (Optional[torch.Generator], optional): Optional random number generator. Defaults to None.
        image_path (str, optional): Path to the image for inpainting. Defaults to "".

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: (final_image, intermediate_images).
    """
    if method == "ddim":
        return ddim_inference(
            unet,
            scheduler_data,
            inference_config.num_steps,
            inference_config.eta,
            inference_config.seed,
            generator,
        )
    elif method == "ddpm":
        return ddpm_inference(
            unet,
            scheduler_data,
            inference_config.num_steps,
            inference_config.seed,
            generator,
        )
    elif method == "inpaint":
        original_image = read_and_format_image(image_path, device=config.device)
        mask = get_mask(original_image)
        return ddpm_inference(
            unet,
            scheduler_data,
            inference_config.num_steps,
            inference_config.seed,
            generator,
            original_image=original_image,
            mask=mask,
        )


def main() -> None:
    """Main execution function."""
    try:
        # torch.use_deterministic_algorithms(True)
        global config
        # Parse command-line arguments
        args = parse_args()

        if args.experiment == "inpaint" and args.image_path == "":
            raise argparse.ArgumentError(
                None, "--image_path is required if --experiment is inpaint."
            )

        # Create config from command-line arguments
        config = set_config(
            experiment=args.experiment,
            dataset=args.dataset,
            save_dir=Path(args.save_dir),
            seed=args.seed,
            image_path=args.image_path,
        )

        logger.info(f"Running {config.experiment} experiment...")

        # Set seed for reproducibility
        SEED = config.seed
        logger.info(f"Using Seed {SEED}: ...")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Create a single generator for the entire process
        generator = torch.Generator(device=config.device).manual_seed(SEED)

        logger.info(f"Loading {config.model_type} model...")
        unet = model_factory(
            model_type=config.model_type,
            model_id=config.model_id,
            weights_path=config.weights_path,
            device=config.device,
        )

        scheduler_params = SchedulerParams()
        scheduler_data = scheduler_params.build_inference_scheduler_parameters()

        # Common configuration
        experiment_config = get_experiment_config(config.experiment)

        # Run and Save DDIM Images
        logger.info(f"Running {config.experiment.upper()} inference...")
        with torch.no_grad():
            _, samples = run_inference(
                config.experiment,
                unet,
                scheduler_data,
                experiment_config,
                generator=generator,
                image_path=config.image_path,
            )
        save_images(
            samples[-1:],
            f"{config.experiment}_steps{experiment_config.num_steps}_seed{config.seed}",
        )

        logger.info("Process completed successfully!")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
