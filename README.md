# DeepFake Detection Multi-Agent System

## Overview

This project implements a multi-agent system for deepfake detection, analysis, and summarization. It uses a combination of deep learning models for detection and a multi-agent architecture powered by LangGraph and ChatGroq to provide comprehensive analysis and human-readable summaries of the results.

## Features

- **Deepfake Detection**: Uses a pre-trained deep learning model to analyze videos and detect potential deepfakes
- **Multi-Agent Analysis**: Employs specialized agents for detection, analysis, and summarization
- **Multiple Interfaces**: Provides command-line, API, Flask web application, and Streamlit web interfaces
- **Detailed Reports**: Generates comprehensive reports with technical details and human-readable summaries

## Requirements

- Python 3.8+
- PyTorch and TorchVision
- LangChain and LangGraph
- Groq API key for LLM access
- Other dependencies listed in `requirements.txt`

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Groq API key (for LLM access)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. Run the setup script:
   ```bash
   ./run.sh setup
   ```

   This will:
   - Check Python version compatibility
   - Install required dependencies
   - Create a `.env` file for your Groq API key
   - Verify the presence of the deepfake detector checkpoint

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key to the `.env` file

## Usage

### Command Line Interface

To analyze a video file:

```bash
./run.sh detect path/to/video.mp4
```

This will output the detection results to the console and optionally save them to a JSON file.

### FastAPI Interface

To start the FastAPI server:

```bash
./run.sh api --port 8000
```

Then you can use the API endpoint:

```bash
curl -X POST -F "video=@/path/to/video.mp4" http://localhost:8000/detect/
```

### Flask Web Application

To start the Flask web application:

```bash
./run.sh flask --port 5000 --debug
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

### Streamlit Web Interface

To start the Streamlit web interface:

```bash
./run.sh web --port 8501
```

Then open your browser and navigate to `http://localhost:8501` to access the web interface.

## Project Structure

- `main.py`: Main entry point for the application
- `model_loader.py`: Handles loading the deepfake detection model
- `video_processor.py`: Processes video files for analysis
- `deepfake_detector.py`: Implements the deepfake detection logic
- `agents.py`: Defines the multi-agent system using LangGraph
- `api.py`: Implements the FastAPI interface
- `flask_app.py`: Implements the Flask web application
- `web_interface.py`: Implements the Streamlit web interface
- `setup.py`: Setup script for installing dependencies
- `test.py`: Tests for verifying system functionality
- `example.py`: Example usage of the system
- `run.sh`: Shell script for running different components

## Model Details

The system uses a pre-trained deep learning model for deepfake detection. The model is based on a convolutional neural network architecture and has been trained on a large dataset of real and fake videos.

The model checkpoint is stored in the file `deepfake_detector_checkpoint.pth` and is loaded automatically by the system.

## Input/Output Format

### Input

The system accepts video files in various formats (mp4, avi, mov, wmv, mkv).

### Output

The system produces a JSON output with the following structure:

```json
{
  "verdict": "REAL" or "FAKE",
  "analysis": "Detailed analysis of the video...",
  "summary": "Summary of findings...",
  "technical_details": {
    "confidence_score": 0.95,
    "frame_predictions": [...],
    "prediction_distribution": {...}
  }
}
```

## Multi-Agent Architecture

The system uses a multi-agent architecture with the following components:

1. **Detector Agent**: Analyzes the video and determines if it's a deepfake
2. **Analysis Agent**: Provides detailed technical analysis of the detection results
3. **Summary Agent**: Generates a human-readable summary of the findings

The agents are orchestrated using LangGraph, which manages the flow of information between them.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- LangChain and LangGraph for the multi-agent framework
- Groq for providing the LLM API
- PyTorch for the deep learning framework