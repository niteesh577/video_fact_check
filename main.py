import os
import json
import argparse
import logging
from typing import Dict, Any


print("→ CWD:", os.getcwd())
print("→ Listing at /Users/mac/Desktop/final_deepfake_detect:", 
      os.listdir("/Users/mac/Desktop/final_deepfake_detect"))
print("→ Exists? ", 
      os.path.exists("/deepfake_detector_checkpoint.pth"))

# Import our custom modules
from deepfake_detector import DeepfakeDetector
from agents import create_agent_workflow, AgentState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the checkpoint file
CHECKPOINT_PATH = "/Users/mac/Desktop/final_deepfake_detect/deepfake_detector_checkpoint.pth"

def process_video(video_path: str) -> Dict[str, Any]:
    """Process a video through the multi-agent deepfake detection system.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: The final output with verdict, analysis, and summary
    """
    try:
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Initialize the deepfake detector
        detector = DeepfakeDetector(CHECKPOINT_PATH)
        
        # Step 2: Analyze the video
        # The new analyze_video returns a dict like:
        # {
        #     'verdict': 'deepfake' or 'real',
        #     'confidence': avg_probability (P(deepfake)),
        #     'frame_analysis': { ... details ... },
        #     'error_message': None or str
        # }
        logger.info(f"Starting deepfake analysis for video: {video_path}")
        detection_results_full = detector.analyze_video(video_path)
        logger.info(f"Deepfake analysis complete. Results: {detection_results_full}")

        # Handle potential errors from analyze_video itself
        if detection_results_full.get('error_message'):
            logger.error(f"Error from analyze_video: {detection_results_full['error_message']}")
            return {
                "verdict": "error",
                "analysis": f"Video analysis failed: {detection_results_full['error_message']}",
                "summary": "Processing failed during video analysis."
            }
        
        # Step 3: Initialize the multi-agent workflow
        workflow = create_agent_workflow()
        
        # Step 4: Prepare the initial state
        # Adapt AgentState initialization to the new detection_results_full structure
        initial_state = AgentState(
            video_path=video_path,
            frames_data=None,  # Assuming frames_data is still not directly needed by agents
            detection_result=detection_results_full, # Pass the entire new dictionary
            analysis="", # To be filled by agents
            verdict=detection_results_full.get('verdict', 'uncertain'), # Extract verdict
            final_output={} # This is typically initialized empty and filled by the graph
        )
        
        # Step 5: Run the workflow
        logger.info("Running multi-agent workflow...")
        final_state = workflow.invoke(initial_state)
        
        # Step 6: Return the final output
        return final_state["final_output"]
    
    except FileNotFoundError as e:
        logger.error(f"Error processing video (FileNotFoundError): {str(e)}")
        return {"verdict": "error", "analysis": f"File not found: {str(e)}", "summary": "Processing failed due to missing file."}
    except IOError as e:
        logger.error(f"Error processing video (IOError): {str(e)}")
        return {"verdict": "error", "analysis": f"IO error: {str(e)}", "summary": "Processing failed due to IO error."}
    except ValueError as e:
        logger.error(f"Error processing video (ValueError): {str(e)}")
        return {"verdict": "error", "analysis": f"Value error: {str(e)}", "summary": "Processing failed due to invalid value or configuration."}
    except RuntimeError as e:
        logger.error(f"Error processing video (RuntimeError): {str(e)}")
        return {"verdict": "error", "analysis": f"Runtime error: {str(e)}", "summary": "Processing failed due to a runtime error."}
    except Exception as e:
        logger.error(f"Unexpected error processing video: {str(e)}", exc_info=True)
        return {
            "verdict": "error",
            "analysis": f"An unexpected error occurred: {str(e)}",
            "summary": "Processing failed due to an unexpected error."
        }

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deepfake Detection Multi-Agent System")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to analyze")
    parser.add_argument("--output", type=str, help="Path to save the output JSON file")
    args = parser.parse_args()
    
    # Check if the video file exists
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Process the video
    result = process_video(args.video)
    
    # Print the result
    print(json.dumps(result, indent=2))
    
    # Save the result to a file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Output saved to {args.output}")

def api_process_video(video_path: str) -> Dict[str, str]:
    """API function to process a video and return the results.
    
    This function can be used as an API endpoint in a web service.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: The final output with verdict, analysis, and summary
    """
    try:
        # Process the video
        result = process_video(video_path)
        return result
    
    except FileNotFoundError as e:
        logger.error(f"API error (FileNotFoundError): {str(e)}")
        return {"verdict": "error", "analysis": f"File not found: {str(e)}", "summary": "Processing failed due to missing file."}
    except IOError as e:
        logger.error(f"API error (IOError): {str(e)}")
        return {"verdict": "error", "analysis": f"IO error: {str(e)}", "summary": "Processing failed due to IO error."}
    except ValueError as e:
        logger.error(f"API error (ValueError): {str(e)}")
        return {"verdict": "error", "analysis": f"Value error: {str(e)}", "summary": "Processing failed due to invalid value or configuration."}
    except RuntimeError as e:
        logger.error(f"API error (RuntimeError): {str(e)}")
        return {"verdict": "error", "analysis": f"Runtime error: {str(e)}", "summary": "Processing failed due to a runtime error."}
    except Exception as e:
        logger.error(f"Unexpected API error: {str(e)}", exc_info=True)
        return {
            "verdict": "error",
            "analysis": f"An unexpected error occurred: {str(e)}",
            "summary": "Processing failed due to an unexpected error."
        }

if __name__ == "__main__":
    main()