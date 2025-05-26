import os
import json
from typing import Dict, List, Any, TypedDict, Annotated, Literal

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_agent_executor

# Deepfake detection model imports
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check if GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in a .env file or export it.")

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Using Llama model as specified
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the state schema for our multi-agent system
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    video_path: str
    detection_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    summary_result: Dict[str, Any]
    final_output: Dict[str, Any]

# Path to the deepfake detection checkpoint
CHECKPOINT_PATH = "/Users/mac/Desktop/final_deepfake_detect/deepfake_detector_checkpoint.pth"

# Load the deepfake detection model
def load_deepfake_detection_model():
    # Note: This is a placeholder implementation that will need to be adjusted
    # based on the actual model architecture used in the checkpoint
    try:
        # Load the checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        
        # Print checkpoint keys for debugging
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
        
        # Assuming the checkpoint contains the model state_dict
        # You'll need to initialize your model with the correct architecture first
        # model = YourModelArchitecture()
        # model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        # model.eval()
        # return model
        
        # For now, return the checkpoint itself for further inspection
        return checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Tool for deepfake detection
@tool
def detect_deepfake(video_path: str) -> Dict[str, Any]:
    """Detect if a video is a deepfake using the pre-trained model."""
    # This is a placeholder implementation
    # In a real implementation, you would:
    # 1. Extract frames from the video
    # 2. Preprocess the frames
    # 3. Run the model on the frames
    # 4. Aggregate the results
    
    try:
        # Load the model (or checkpoint for now)
        model_or_checkpoint = load_deepfake_detection_model()
        
        if model_or_checkpoint is None:
            return {"error": "Failed to load the deepfake detection model"}
        
        # For demonstration purposes, let's extract a few frames from the video
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        max_frames = 10  # Process up to 10 frames
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
        cap.release()
        
        if not frames:
            return {"error": "Could not extract frames from the video"}
        
        # For now, return a placeholder result
        # In a real implementation, you would process these frames with your model
        return {
            "verdict": "deepfake",  # This would be the actual model prediction
            "confidence": 0.85,     # This would be the model's confidence score
            "processed_frames": len(frames),
            "model_loaded": True
        }
        
    except Exception as e:
        return {"error": f"Error in deepfake detection: {str(e)}"}

# Define the agents

# 1. Deepfake Detection Agent
detection_system_prompt = """
You are a specialized Deepfake Detection Agent. Your task is to analyze videos to determine if they are deepfakes or authentic.

You have access to a state-of-the-art deepfake detection model that has been trained to identify manipulated videos.

When given a video, you should:
1. Use the deepfake detection tool to analyze the video
2. Interpret the results from the model
3. Provide a clear verdict on whether the video is a deepfake or authentic
4. Include the confidence level of the detection

Be precise and factual in your assessment.
"""

detection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=detection_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="Please analyze this video: {video_path}"),
    ]
)

detection_agent = create_agent_executor(
    llm=llm,
    tools=[detect_deepfake],
    prompt=detection_prompt,
    verbose=True
)

# 2. Analysis Agent
analysis_system_prompt = """
You are a specialized Deepfake Analysis Agent. Your task is to provide detailed analysis of videos that have been flagged as potential deepfakes.

Based on the detection results provided, you should:
1. Explain the technical indicators that suggest the video is a deepfake or authentic
2. Identify specific artifacts or inconsistencies if it's a deepfake
3. Assess the quality and sophistication of the deepfake (if applicable)
4. Provide context on the type of deepfake technique likely used
5. Discuss potential implications of such a deepfake

Be thorough and educational in your analysis.
"""

analysis_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=analysis_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="Please analyze the following detection results: {detection_result}"),
    ]
)

analysis_agent = create_agent_executor(
    llm=llm,
    prompt=analysis_prompt,
    verbose=True
)

# 3. Summary Agent
summary_system_prompt = """
You are a specialized Deepfake Summary Agent. Your task is to create concise, informative summaries of deepfake detection and analysis results.

Based on the detection and analysis provided, you should:
1. Summarize the key findings in clear, non-technical language
2. Highlight the most important aspects of the analysis
3. Provide a balanced assessment of the confidence in the results
4. Include any relevant context or implications

Your summary should be accessible to non-experts while still being accurate and informative.
"""

summary_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=summary_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="Please summarize the following detection and analysis results:\n\nDetection: {detection_result}\n\nAnalysis: {analysis_result}"),
    ]
)

summary_agent = create_agent_executor(
    llm=llm,
    prompt=summary_prompt,
    verbose=True
)

# Define the supervisor agent
supervisor_system_prompt = """
You are the Supervisor Agent coordinating a multi-agent system for deepfake detection and analysis.

Your responsibilities include:
1. Orchestrating the workflow between specialized agents
2. Determining which agent should act next based on the current state
3. Ensuring all necessary information is passed between agents
4. Compiling the final output from all agents' contributions

The agents you supervise are:
- Deepfake Detection Agent: Determines if a video is a deepfake
- Analysis Agent: Provides detailed technical analysis of the detection results
- Summary Agent: Creates a concise, informative summary of the findings

Your goal is to ensure a comprehensive and accurate assessment of the video.
"""

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="""Current state: {state}

Based on this state, which agent should act next? Choose from:
- 'detection': The Deepfake Detection Agent should analyze the video
- 'analysis': The Analysis Agent should examine the detection results
- 'summary': The Summary Agent should summarize the findings
- 'finish': All agents have completed their tasks, and we should compile the final output

Respond with just one of these options."""),
    ]
)

# Function to determine the next agent
def determine_next_agent(state: AgentState) -> Literal["detection", "analysis", "summary", "finish"]:
    # Get the supervisor's decision
    messages = state.get("messages", [])
    
    # Create a simplified state representation for the supervisor
    state_for_supervisor = {
        "video_path": state.get("video_path", "Not provided"),
        "detection_result": state.get("detection_result", {}),
        "analysis_result": state.get("analysis_result", {}),
        "summary_result": state.get("summary_result", {}),
    }
    
    response = supervisor_prompt.invoke({"messages": messages, "state": json.dumps(state_for_supervisor)})
    supervisor_decision = llm.invoke([response]).content.strip().lower()
    
    # Map the supervisor's decision to the next agent
    if "detection" in supervisor_decision:
        return "detection"
    elif "analysis" in supervisor_decision:
        return "analysis"
    elif "summary" in supervisor_decision:
        return "summary"
    else:
        return "finish"

# Define the agent functions

def run_detection_agent(state: AgentState) -> AgentState:
    """Run the deepfake detection agent."""
    video_path = state["video_path"]
    messages = state.get("messages", [])
    
    # Invoke the detection agent
    response = detection_agent.invoke({"messages": messages, "video_path": video_path})
    
    # Extract the detection result from the agent's response
    detection_result = {}
    for message in response["messages"]:
        if message["type"] == "function" and message["name"] == "detect_deepfake":
            detection_result = json.loads(message["content"])
            break
    
    # If no function message was found, use the last AI message
    if not detection_result:
        for message in reversed(response["messages"]):
            if message["type"] == "ai":
                # Try to parse the AI message as JSON
                try:
                    detection_result = json.loads(message["content"])
                except:
                    detection_result = {"verdict": message["content"]}
                break
    
    # Update the state
    return {**state, "detection_result": detection_result, "messages": response["messages"]}

def run_analysis_agent(state: AgentState) -> AgentState:
    """Run the analysis agent."""
    detection_result = state["detection_result"]
    messages = state.get("messages", [])
    
    # Invoke the analysis agent
    response = analysis_agent.invoke({"messages": messages, "detection_result": json.dumps(detection_result)})
    
    # Extract the analysis result from the agent's response
    analysis_result = {}
    for message in reversed(response["messages"]):
        if message["type"] == "ai":
            analysis_result = {"analysis": message["content"]}
            break
    
    # Update the state
    return {**state, "analysis_result": analysis_result, "messages": response["messages"]}

def run_summary_agent(state: AgentState) -> AgentState:
    """Run the summary agent."""
    detection_result = state["detection_result"]
    analysis_result = state["analysis_result"]
    messages = state.get("messages", [])
    
    # Invoke the summary agent
    response = summary_agent.invoke({
        "messages": messages, 
        "detection_result": json.dumps(detection_result),
        "analysis_result": json.dumps(analysis_result)
    })
    
    # Extract the summary result from the agent's response
    summary_result = {}
    for message in reversed(response["messages"]):
        if message["type"] == "ai":
            summary_result = {"summary": message["content"]}
            break
    
    # Update the state
    return {**state, "summary_result": summary_result, "messages": response["messages"]}

def compile_final_output(state: AgentState) -> AgentState:
    """Compile the final output from all agents' results."""
    detection_result = state["detection_result"]
    analysis_result = state["analysis_result"]
    summary_result = state["summary_result"]
    
    # Create the final output
    final_output = {
        "verdict": detection_result.get("verdict", "Unknown"),
        "analysis": analysis_result.get("analysis", "No analysis provided"),
        "summary": summary_result.get("summary", "No summary provided")
    }
    
    # Update the state
    return {**state, "final_output": final_output}

workflow = StateGraph(AgentState)

# ENTRYPOINT: wire the built-in START to your first node
workflow.add_edge(START, "detection")

# Add your processing nodes
workflow.add_node("detection", run_detection_agent)
workflow.add_node("analysis",  run_analysis_agent)
workflow.add_node("summary",   run_summary_agent)
workflow.add_node("finish",    compile_final_output)

# Conditional transitions
workflow.add_conditional_edges("detection", determine_next_agent, {
    "analysis": "analysis", "summary": "summary", "finish": "finish", "detection": "detection"
})
workflow.add_conditional_edges("analysis",  determine_next_agent, {
    "analysis": "analysis", "summary": "summary", "finish": "finish", "detection": "detection"
})
workflow.add_conditional_edges("summary",   determine_next_agent, {
    "analysis": "analysis", "summary": "summary", "finish": "finish", "detection": "detection"
})

# End of workflow
workflow.add_edge("finish", END)

# Compile once
app = workflow.compile()

# Function to process a video
def process_video(video_path: str) -> Dict[str, Any]:
    """Process a video through the multi-agent system."""
    # Initialize the state
    initial_state = {
        "messages": [],
        "video_path": video_path,
        "detection_result": {},
        "analysis_result": {},
        "summary_result": {},
        "final_output": {}
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Return the final output
    return result["final_output"]

# Example usage
if __name__ == "__main__":
    # Replace with the path to your video
    video_path = "/path/to/your/video.mp4"
    
    # Process the video
    result = process_video(video_path)
    
    # Print the result
    print(json.dumps(result, indent=2))