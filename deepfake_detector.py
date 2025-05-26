import os
import torch
import numpy as np
import logging
from typing import Dict, Any, List

from model import FusionDeepfakeDetector # Use the model from model.py
# VideoProcessor might not be needed if all processing is handled here
# from video_processor import VideoProcessor
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from torchvision import transforms as T
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Class for detecting deepfakes in videos using FusionDeepfakeDetector."""

    def __init__(self, checkpoint_path: str):
        # The user's code hardcodes the checkpoint path inside the loading logic.
        # We'll keep checkpoint_path as an argument for flexibility, but the user's
        # provided loading logic might override it or expect a specific name.
        self.checkpoint_path = checkpoint_path # This might be '/content/deepfake_detector_checkpoint.pth copy'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize MTCNN for face detection (as per user's code)
        self.mtcnn = MTCNN(
            image_size=224, margin=14, keep_all=False, # keep_all=False to get only the best face
            post_process=True, # Ensure output is a tensor for ToPILImage
            device=self.device
        )
        logger.info("MTCNN initialized.")

        # Define transformations (must match training transforms, as per user's code)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Transforms defined.")

        self._load_model()

    def _load_model(self):
        """Load the FusionDeepfakeDetector model."""
        try:
            # Use the user-provided checkpoint path and loading mechanism
            # The user's example uses '/content/deepfake_detector_checkpoint.pth copy'
            # For robustness, we'll use self.checkpoint_path if it's a real file, 
            # otherwise, we fall back to the user's hardcoded path as a last resort.
            effective_checkpoint_path = self.checkpoint_path
            if not os.path.exists(effective_checkpoint_path):
                logger.warning(f"Checkpoint at '{effective_checkpoint_path}' not found. Trying user's hardcoded path.")
                # This is the path from the user's example script
                # THIS IS A TEMPORARY WORKAROUND. The user should fix their checkpoint path.
                # effective_checkpoint_path = '/content/deepfake_detector_checkpoint.pth copy'
                # For now, let's assume the provided self.checkpoint_path is the correct one and error if not found.
                # If the error persists, the user needs to ensure deepfake_detector_checkpoint.pth is at the specified location.
                pass # Let it fail below if self.checkpoint_path is also wrong

            logger.info(f"Attempting to load model from checkpoint: {effective_checkpoint_path}")
            if not os.path.exists(effective_checkpoint_path):
                # This is the critical error the user reported.
                error_msg = f"Checkpoint file not found at {effective_checkpoint_path}"
                logger.error(error_msg)
                # To make this error surface in main.py's error handling for analyze_video:
                self.model = None # Indicate model loading failed
                self.model_load_error = error_msg
                return

            self.model = FusionDeepfakeDetector().to(self.device)
            checkpoint = torch.load(effective_checkpoint_path, map_location=self.device)
            
            # Check for different possible key names for the state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assuming the checkpoint itself is the state_dict
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_load_error = None # Clear error on success
            logger.info(f"FusionDeepfakeDetector model loaded successfully from {effective_checkpoint_path} on {self.device}")
        except FileNotFoundError as e:
            logger.error(f"Checkpoint file not found: {e}")
            self.model = None
            self.model_load_error = str(e)
        except Exception as e:
            logger.error(f"Error loading FusionDeepfakeDetector model: {e}", exc_info=True)
            self.model = None # Indicate model loading failed
            self.model_load_error = str(e)
            # raise # Re-raising might hide the specific error message in some contexts

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Predicts the deepfake probability for a given video using FusionDeepfakeDetector."""
        if hasattr(self, 'model_load_error') and self.model_load_error:
            logger.error(f"Model not loaded due to error: {self.model_load_error}")
            return {
                "verdict": "error",
                "confidence": 0.0,
                "details": {},
                "error_message": f"Model loading failed: {self.model_load_error}"
            }
        if self.model is None:
            # This case should ideally be caught by the above, but as a safeguard:
            logger.error("Model is not loaded. Cannot analyze video.")
            return {
                "verdict": "error", 
                "confidence": 0.0, 
                "details": {},
                "error_message": "Model is not loaded."
            }

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"verdict": "error", "confidence": 0.0, "details": {}, "error_message": f"Video file not found: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return {"verdict": "error", "confidence": 0.0, "details": {}, "error_message": f"Cannot open video file: {video_path}"}

        predictions: List[float] = []
        frames_processed_with_faces = 0
        frames_read_count = 0
        max_frames_to_process = 20  # As per user's example logic

        try:
            with torch.no_grad():
                for i in tqdm(range(max_frames_to_process), desc="Processing video frames"):
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video or cannot read frame.")
                        break
                    frames_read_count += 1

                    # Convert frame to PIL Image
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Detect face and crop using MTCNN
                    # MTCNN (post_process=True) returns a batch of face tensors or None
                    face_batch = self.mtcnn(img_pil)

                    if face_batch is not None:
                        # Assuming we take the first detected face if multiple
                        face_tensor = face_batch[0] if isinstance(face_batch, list) else face_batch 
                        if face_tensor.ndim == 4 and face_tensor.shape[0] == 1: # If MTCNN returns a batch of 1
                            face_tensor = face_tensor.squeeze(0)
                        
                        # The user's code converts tensor to PIL then transforms.
                        # If MTCNN output (with post_process=True) is already a normalized tensor (0-1 range),
                        # ToPILImage might not be strictly necessary if transform expects tensor.
                        # However, to match user's logic:
                        face_pil_from_tensor = T.ToPILImage()(face_tensor.cpu()) # MTCNN output is on device
                        
                        img_transformed = self.transform(face_pil_from_tensor).unsqueeze(0).to(self.device)
                        
                        # The user's FusionDeepfakeDetector's forward method returns sigmoid output already.
                        prob = self.model(img_transformed).item() # model output is [B], so .item()
                        predictions.append(prob)
                        frames_processed_with_faces += 1
                    else:
                        logger.debug(f"No face detected in frame {frames_read_count}")
        except Exception as e:
            logger.error(f"Error during video processing: {e}", exc_info=True)
            cap.release()
            return {"verdict": "error", "confidence": 0.0, "details": {}, "error_message": f"Runtime error during processing: {e}"}
        finally:
            cap.release()

        avg_probability = 0.5  # Default if no predictions
        if predictions:
            avg_probability = sum(predictions) / len(predictions)
        else:
            logger.warning("No faces detected in any processed frames. Defaulting to neutral probability.")

        # Determine verdict based on threshold (user's example: 0.52, deepfake if prob > threshold)
        # The user's example says: if probability > threshold: print("Video is likely real.") else: print("Video is likely a deepfake.")
        # This implies higher probability = real. Let's clarify this. Assuming P(deepfake).
        # If model outputs P(deepfake):
        #   P(deepfake) > threshold => deepfake
        #   P(deepfake) <= threshold => real
        # The user's example code has: `if probability > threshold: print("Video is likely real.")`
        # This means their `probability` is P(real). So, if our model outputs P(deepfake), we need to adjust.
        # The FusionDeepfakeDetector's sigmoid outputs P(deepfake) typically.
        # Let's assume avg_probability is P(deepfake).
        threshold = 0.5 # A common threshold for P(deepfake)
        verdict = "deepfake" if avg_probability > threshold else "real"
        
        # If the user's interpretation (higher prob = real) is strict, then:
        # verdict = "real" if avg_probability > threshold else "deepfake"
        # For now, sticking to standard P(deepfake) > threshold = deepfake.

        result = {
            "verdict": verdict,
            "confidence": avg_probability, # This is P(deepfake)
            "details": {
                "frames_processed_with_faces": frames_processed_with_faces,
                "frames_read": frames_read_count,
                "average_deepfake_probability": avg_probability,
                "frame_probabilities": predictions,
                "threshold_used": threshold
            },
            "error_message": None
        }
        logger.info(f"Video analysis complete. Verdict: {verdict}, Confidence (P(deepfake)): {avg_probability:.4f}")
        return result

    def _analyze_consistency(self, preds: List[str]) -> Dict[str, Any]:
        """Analyze consistency across frame-level predictions."""
        transitions = sum(
            1 for i in range(1, len(preds)) if preds[i] != preds[i-1]
        )
        score = 1.0 - transitions / max(1, len(preds)-1)
        return {
            "score": score,
            "transitions": transitions,
            "interpretation": (
                "high" if score>0.8 else "medium" if score>0.5 else "low"
            )
        }

    def _analyze_confidence(self, confs: List[float]) -> Dict[str, float]:
        arr = np.array(confs)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std())
        }


# Example usage
def main():
    # Path to the checkpoint file
    checkpoint_path = "/Users/mac/Desktop/final_deepfake_detect/deepfake_detector_checkpoint.pth"
    
    # Initialize the detector
    detector = DeepfakeDetector(checkpoint_path)
    
    # Path to a video file
    video_path = "path/to/your/video.mp4"
    
    # Analyze the video
    result = detector.analyze_video(video_path)
    
    # Print the result
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Frame predictions: {result['frame_predictions']}")
    print(f"Consistency analysis: {result['analysis']['consistency']}")
    print(f"Confidence distribution: {result['analysis']['confidence_distribution']}")

if __name__ == "__main__":
    main()