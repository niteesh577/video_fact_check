import torch
import torch.nn as nn
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new model
from model import FusionDeepfakeDetector

def load_model(checkpoint_path):
    """Load the deepfake detection model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file
        
    Returns:
        model: The loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Create model instance
        model = FusionDeepfakeDetector().to(device)
        
        # Load checkpoint
        logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load the model state dictionary
        # Assuming the checkpoint is a dictionary with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model_state_dict from checkpoint.")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # For compatibility with older format
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded state_dict from checkpoint.")
        else:
            # If the checkpoint is directly the state_dict (less common for complex models)
            model.load_state_dict(checkpoint)
            logger.info("Loaded checkpoint directly as state_dict.")
        
        model.eval()  # Set model to evaluation mode
        logger.info("Model loaded successfully and set to evaluation mode.")
        return model, device
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict(model, image_tensor, device):
    """Make a prediction using the loaded FusionDeepfakeDetector model.
    
    Args:
        model: The loaded FusionDeepfakeDetector model.
        image_tensor: The preprocessed input image tensor (already transformed and on the correct device).
        device: The device the model and tensor are on.
        
    Returns:
        float: Probability of the image being a deepfake (as per FusionDeepfakeDetector output).
    """
    try:
        with torch.no_grad():
            # Ensure image_tensor is on the correct device and has batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            logit = model(image_tensor)
            probability = torch.sigmoid(logit).item() # Model directly outputs logit, apply sigmoid
            return probability
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise