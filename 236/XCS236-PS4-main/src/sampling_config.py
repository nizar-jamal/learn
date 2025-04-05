from dataclasses import dataclass
from pathlib import Path
import logging
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    experiment: str
    dataset: str
    model_type: str = "unet"
    model_id: str = "google/ddpm-celebahq-256"
    weights_path: str = "./weights_epoch_8_t1000_img32.pth"
    image_scale: int = 255
    save_dir: Path = str
    device: str = "cpu"
    seed: int = 42
    image_path: str = ""

    def __post_init__(self):
        if self.save_dir == Path(""):
            self.save_dir = Path(f"{self.dataset}_samples")
        if self.dataset == "mnist":
            self.model_type = "unet"
            self.device = "cpu"
        elif self.dataset == "faces":
            self.model_type = "unet2d"
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            raise ValueError("Invalid dataset type. Choose 'mnist' or 'faces'.")

        logger.info(f"Using device: {self.device}")

        valid_devices = ["cuda", "mps", "cpu"]
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device {self.device}. Must be one of {valid_devices}"
            )

# Singleton instance
_config_instance = None

def get_config(**kwargs):
    global _config_instance
    if _config_instance is None:
        # _config_instance = Config(
        #     **kwargs
        # )
        pass
    return _config_instance

def set_config(**kwargs):
    global _config_instance
    _config_instance = Config(**kwargs)
    return _config_instance