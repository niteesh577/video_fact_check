import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Class for processing video files for deepfake detection."""
    
    def __init__(self, frame_count=30):
        """Initialize the video processor.
        
        Args:
            frame_count (int): Number of frames to extract from the video
        """
        self.frame_count = frame_count # Should align with DeepfakeDetector's usage, e.g., 20
        # Transformation is now handled in DeepfakeDetector after MTCNN processing.
        # This class will focus on raw frame extraction.
        # self.transform = transforms.Compose([...]) # Removed from here
    
    def extract_frames(self, video_path):
        """Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of extracted frames as tensors
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        
        logger.info(f"Extracting frames from video: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip = max(1, frame_count // self.frame_count)
            
            for i in range(0, frame_count, skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image (or return raw cv2 frame if preferred by DeepfakeDetector)
                # DeepfakeDetector expects PIL images for MTCNN
                pil_image = Image.fromarray(frame)
                frames.append(pil_image) # Now returns list of PIL Images
                
                if len(frames) >= self.frame_count:
                    break
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def process_video(self, video_path):
        """Extract frames from a video for deepfake detection.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of PIL.Image objects representing the frames.
        """
        # extract_frames now returns a list of PIL Images
        frames = self.extract_frames(video_path)
        return frames
    
    def get_sample_frame(self, video_path):
        """Extract a single sample frame from the video for visualization.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            numpy.ndarray: Sample frame as a numpy array (RGB)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get the middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError(f"Could not read frame from {video_path}")
            
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        except Exception as e:
            logger.error(f"Error getting sample frame: {str(e)}")
            raise