#!/bin/bash

# Run script for DeepFake Detection Multi-Agent System

# Function to display help message
show_help() {
    echo "DeepFake Detection Multi-Agent System"
    echo ""
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup              Run the setup script to install dependencies and configure the environment"
    echo "  test [--video]     Run tests to verify that all components are working correctly"
    echo "  api                Start the FastAPI server"
    echo "  flask              Start the Flask web application"
    echo "  web                Start the Streamlit web interface"
    echo "  detect [video]     Analyze a video for deepfakes"
    echo "  help               Show this help message"
    echo ""
    echo "Options:"
    echo "  --video PATH       Path to a video file (for test command)"
    echo "  --port PORT        Port to run the API or web interface on (default: 8000 for API, 5000 for Flask, 8501 for Streamlit)"
    echo "  --host HOST        Host to run the server on (default: 0.0.0.0)"
    echo "  --debug            Run in debug mode (for Flask app)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh test --video path/to/video.mp4"
    echo "  ./run.sh api --port 8080"
    echo "  ./run.sh flask --port 5000 --debug"
    echo "  ./run.sh web --port 8501"
    echo "  ./run.sh detect path/to/video.mp4"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
}

# Parse command
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

# Process command
case $COMMAND in
    setup)
        echo "Running setup..."
        python3 setup.py "$@"
        ;;
    test)
        echo "Running tests..."
        python3 test.py "$@"
        ;;
    api)
        echo "Starting API server..."
        PORT=8000
        HOST="0.0.0.0"
        # Parse port option
        while [[ $# -gt 0 ]]; do
            case $1 in
                --port)
                    PORT="$2"
                    shift 2
                    ;;
                --host)
                    HOST="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        python3 api.py --port $PORT --host $HOST
        ;;
    flask)
        echo "Starting Flask web application..."
        PORT=5000
        HOST="0.0.0.0"
        DEBUG=""
        # Parse options
        while [[ $# -gt 0 ]]; do
            case $1 in
                --port)
                    PORT="$2"
                    shift 2
                    ;;
                --host)
                    HOST="$2"
                    shift 2
                    ;;
                --debug)
                    DEBUG="--debug"
                    shift
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        python3 flask_app.py --port $PORT --host $HOST $DEBUG
        ;;
    web)
        echo "Starting web interface..."
        PORT=8501
        # Parse port option
        while [[ $# -gt 0 ]]; do
            case $1 in
                --port)
                    PORT="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        streamlit run web_interface.py --server.port $PORT
        ;;
    detect)
        echo "Analyzing video for deepfakes..."
        if [ $# -eq 0 ]; then
            echo "Error: No video file specified."
            echo "Usage: ./run.sh detect path/to/video.mp4"
            exit 1
        fi
        python3 main.py --video "$1"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_help
        exit 1
        ;;
esac