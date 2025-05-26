from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
import logging
from main import process_video
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepFake Detection API", description="API for detecting deepfake videos using a multi-agent system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "DeepFake Detection API is running"}

@app.post("/detect/")
async def detect_deepfake(video: UploadFile = File(...)):
    """Endpoint to detect if a video is a deepfake.
    
    Args:
        video (UploadFile): The video file to analyze
        
    Returns:
        JSONResponse: Detection results including verdict and analysis
    """
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(video.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the video
            logger.info(f"Processing uploaded video: {video.filename}")
            result = process_video(temp_file_path)
            
            # Return the results
            return JSONResponse(content={
                "filename": video.filename,
                "verdict": result.get("verdict", "unknown"),
                "analysis": result.get("analysis", "No analysis available"),
                "summary": result.get("summary", "No summary available")
            })
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

def start_api(host="0.0.0.0", port=8000):
    """Start the FastAPI server.
    
    Args:
        host (str): Host to bind the server to
        port (int): Port to bind the server to
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api()