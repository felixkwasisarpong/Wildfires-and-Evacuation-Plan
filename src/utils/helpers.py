from config import config
import logging
import numpy as np
import torch

# Setup logging
def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Normalize image data
def normalize_image(image):
    """Normalize input image to range [0, 1]."""
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val)

# Convert torch tensor to numpy image
def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array."""
    return tensor.detach().cpu().numpy()

# Load trained model
def load_model(model):
    """Load model weights if available."""
    model_path = config.paths["model_path"]
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    logger.info(f"Model loaded from {model_path}")
    return model
