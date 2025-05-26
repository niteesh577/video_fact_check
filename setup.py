import os
import argparse
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if the Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required. You have {current_version[0]}.{current_version[1]}.")
        return False
    
    return True

def install_dependencies():
    """Install the required dependencies."""
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def create_env_file(groq_api_key=None):
    """Create the .env file with the necessary environment variables."""
    if not groq_api_key:
        groq_api_key = input("Enter your Groq API key: ")
    
    env_content = f"GROQ_API_KEY={groq_api_key}\n"
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info(".env file created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {str(e)}")
        return False

def check_model_checkpoint():
    """Check if the model checkpoint file exists."""
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_detector_checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Model checkpoint not found at {checkpoint_path}")
        return False
    
    logger.info(f"Model checkpoint found at {checkpoint_path}")
    return True

def setup(groq_api_key=None):
    """Run the complete setup process."""
    logger.info("Starting setup process...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create .env file
    if not create_env_file(groq_api_key):
        return False
    
    # Check model checkpoint
    check_model_checkpoint()
    
    logger.info("Setup completed successfully!")
    return True

def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(description="Setup script for DeepFake Detection Multi-Agent System")
    parser.add_argument("--groq-api-key", type=str, help="Your Groq API key")
    args = parser.parse_args()
    
    setup(args.groq_api_key)

if __name__ == "__main__":
    main()