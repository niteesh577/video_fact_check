import os
import torch
import logging
import argparse
from dotenv import load_dotenv
from model_loader import load_model, DeepFakeDetector
from video_processor import VideoProcessor
from agents import get_llm, create_agent_workflow

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_detector_checkpoint.pth")

def test_model_loading():
    """Test loading the deepfake detection model."""
    try:
        logger.info("Testing model loading...")
        model = load_model(CHECKPOINT_PATH)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def test_video_processor(video_path=None):
    """Test the video processor."""
    if not video_path:
        logger.warning("No video path provided, skipping video processor test.")
        return True
    
    try:
        logger.info(f"Testing video processor with video: {video_path}")
        processor = VideoProcessor()
        frames = processor.extract_frames(video_path)
        logger.info(f"Successfully extracted {len(frames)} frames from video.")
        return True
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return False

def test_llm_connection():
    """Test the connection to the LLM."""
    try:
        logger.info("Testing LLM connection...")
        llm = get_llm()
        response = llm.invoke("Hello, are you working properly?")
        logger.info(f"LLM responded: {response.content[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Error connecting to LLM: {str(e)}")
        return False

def test_agent_workflow():
    """Test creating the agent workflow."""
    try:
        logger.info("Testing agent workflow creation...")
        workflow = create_agent_workflow()
        logger.info("Agent workflow created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating agent workflow: {str(e)}")
        return False

def run_tests(video_path=None):
    """Run all tests."""
    print("\n===== RUNNING TESTS =====")
    
    # Test model loading
    print("\n1. Testing model loading...")
    if test_model_loading():
        print("✅ Model loading test passed!")
    else:
        print("❌ Model loading test failed!")
    
    # Test video processor
    print("\n2. Testing video processor...")
    if test_video_processor(video_path):
        print("✅ Video processor test passed!")
    else:
        print("❌ Video processor test failed!")
    
    # Test LLM connection
    print("\n3. Testing LLM connection...")
    if test_llm_connection():
        print("✅ LLM connection test passed!")
    else:
        print("❌ LLM connection test failed!")
    
    # Test agent workflow
    print("\n4. Testing agent workflow...")
    if test_agent_workflow():
        print("✅ Agent workflow test passed!")
    else:
        print("❌ Agent workflow test failed!")
    
    print("\n===== TEST SUMMARY =====")

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test script for DeepFake Detection Multi-Agent System")
    parser.add_argument("--video", type=str, help="Path to a test video file")
    args = parser.parse_args()
    
    run_tests(args.video)

if __name__ == "__main__":
    main()