import streamlit as st
import os
import tempfile
import json
import time
from dotenv import load_dotenv
from main import process_video
import matplotlib.pyplot as plt
import numpy as np
from video_processor import VideoProcessor

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for the Streamlit web interface."""
    st.title("DeepFake Detection Multi-Agent System")
    st.markdown("""
    This application uses a multi-agent system to detect, analyze, and summarize deepfake videos.
    Upload a video to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Display the video
            st.video(uploaded_file)
            
            # Process button
            if st.button("Analyze Video"):
                with st.spinner("Processing video..."):
                    # Process the video
                    start_time = time.time()
                    result = process_video(temp_file_path)
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.success(f"Processing completed in {processing_time:.2f} seconds!")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs(["Verdict", "Analysis", "Summary", "Technical Details"])
                    
                    with tab1:
                        st.header("Verdict")
                        verdict = result.get("verdict", "unknown")
                        
                        # Display verdict with appropriate styling
                        if verdict == "deepfake":
                            st.error(f"⚠️ This video is a **DEEPFAKE**")
                        elif verdict == "real":
                            st.success(f"✅ This video is **REAL**")
                        else:
                            st.warning(f"❓ The verdict is **UNKNOWN**")
                    
                    with tab2:
                        st.header("Detailed Analysis")
                        st.markdown(result.get("analysis", "No analysis available"))
                    
                    with tab3:
                        st.header("Summary")
                        st.markdown(result.get("summary", "No summary available"))
                    
                    with tab4:
                        st.header("Technical Details")
                        
                        # Try to extract and display frame results if available
                        try:
                            if "detection_result" in result and "frame_results" in result["detection_result"]:
                                frame_results = result["detection_result"]["frame_results"]
                                
                                # Create a dataframe for the results
                                import pandas as pd
                                df = pd.DataFrame(frame_results)
                                st.dataframe(df)
                                
                                # Create a visualization
                                fig, ax = plt.subplots(figsize=(10, 6))
                                df.plot(x="frame_idx", y="confidence", kind="line", ax=ax)
                                ax.set_title("Confidence Scores by Frame")
                                ax.set_xlabel("Frame Index")
                                ax.set_ylabel("Confidence Score")
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Display a sample frame
                                try:
                                    processor = VideoProcessor()
                                    sample_frame = processor.get_sample_frame(temp_file_path)
                                    st.image(sample_frame, caption="Sample Frame from Video", use_column_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display sample frame: {str(e)}")
                            else:
                                st.info("No detailed frame results available.")
                        except Exception as e:
                            st.warning(f"Error displaying technical details: {str(e)}")
                    
                    # Save results button
                    if st.download_button(
                        label="Download Results as JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_results.json",
                        mime="application/json"
                    ):
                        st.success("Results downloaded successfully!")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    # Add information about the system
    st.sidebar.title("About")
    st.sidebar.info("""
    This application uses a multi-agent system powered by LangGraph and ChatGroq with the Llama model to detect, analyze, and summarize deepfake videos.
    
    The system consists of three specialized agents:
    1. **Detector Agent**: Determines if a video is real or fake
    2. **Analysis Agent**: Provides detailed technical analysis
    3. **Summary Agent**: Generates a concise summary
    """)
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    © 2023 DeepFake Detection System
    
    Built with Streamlit, LangGraph, and ChatGroq
    """)

if __name__ == "__main__":
    main()