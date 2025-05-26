import os
import json
from dotenv import load_dotenv
from main import process_video

# Load environment variables
load_dotenv()

def run_example(video_path):
    """Run an example deepfake detection on a video.
    
    Args:
        video_path (str): Path to the video file
    """
    print(f"\n===== PROCESSING VIDEO: {video_path} =====")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Process the video
    result = process_video(video_path)
    
    # Print the results
    print("\n===== DEEPFAKE DETECTION RESULTS =====")
    print(f"Verdict: {result.get('verdict', 'unknown')}")
    print("\n===== DETAILED ANALYSIS =====")
    print(result.get('analysis', 'No analysis available'))
    print("\n===== SUMMARY =====")
    print(result.get('summary', 'No summary available'))
    
    # Save the results to a JSON file
    output_path = f"{os.path.splitext(video_path)[0]}_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

def main():
    """Main function to demonstrate usage."""
    # You can replace this with your own video path
    video_path = input("Enter the path to the video file: ")
    run_example(video_path)

if __name__ == "__main__":
    main()