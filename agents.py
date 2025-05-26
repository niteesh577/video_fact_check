from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Literal
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define state schema
class AgentState(TypedDict):
    """State for the multi-agent system."""
    video_path: str
    frames_data: Any
    detection_result: Dict[str, Any]
    analysis: str
    verdict: Literal["deepfake", "real", "unknown"]
    final_output: Dict[str, str]

# Initialize LLM
def get_llm(model_name="llama-3.1-8b-instant"):
    """Initialize the LLM with the specified model."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        llm = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

# Define agent prompts
detector_prompt = ChatPromptTemplate.from_template("""
    You are a specialized DeepFake Detection Agent. Your task is to analyze the results from 
    a deepfake detection model and determine if the video is real or fake.
    
    Detection model results: {detection_result}
    
    Based on this information, provide your verdict on whether the video is a deepfake or real.
    Your response should be structured as follows:
    
    Verdict: [deepfake/real]
    Confidence: [confidence level]
    Reasoning: [brief explanation of your verdict]
""")

analysis_prompt = ChatPromptTemplate.from_template("""
    You are a specialized Video Analysis Agent. Your task is to analyze a video that has been 
    processed by a deepfake detection system.
    
    Detection results: {detection_result}
    Verdict: {verdict}
    
    Based on this information, provide a detailed analysis of the video. Consider:
    1. What visual artifacts or inconsistencies might be present if it's a deepfake
    2. What technical aspects of the video support the verdict
    3. What potential methods might have been used to create the deepfake (if applicable)
    4. What level of sophistication the deepfake demonstrates (if applicable)
    
    Your analysis should be detailed, technical, and insightful.
""")

summary_prompt = ChatPromptTemplate.from_template("""
    You are a specialized Summary Agent. Your task is to create a concise summary of the 
    deepfake detection and analysis process.
    
    Verdict: {verdict}
    Analysis: {analysis}
    
    Create a comprehensive but concise summary that includes:
    1. The final verdict on whether the video is real or a deepfake
    2. The key points from the analysis
    3. Any recommendations or next steps
    
    Format your response as a well-structured summary that could be presented to a non-technical audience.
""")

# Define agent functions
def detector_agent(state: AgentState) -> AgentState:
    """Agent responsible for determining if a video is deepfake or real."""
    logger.info("Detector agent processing...")
    
    try:
        llm = get_llm()
        chain = detector_prompt | llm
        response = chain.invoke({"detection_result": state["detection_result"]})
        response_text = response.content
        
        # Parse the response to extract verdict and confidence
        lines = response_text.strip().split('\n')
        verdict_line = next((line for line in lines if line.startswith("Verdict:")), "")
        verdict = "unknown"
        
        if "deepfake" in verdict_line.lower():
            verdict = "deepfake"
        elif "real" in verdict_line.lower():
            verdict = "real"
        
        # Update state
        return {**state, "verdict": verdict}
    
    except Exception as e:
        logger.error(f"Error in detector agent: {str(e)}")
        return {**state, "verdict": "unknown"}

def analysis_agent(state: AgentState) -> AgentState:
    """Agent responsible for analyzing the video in detail."""
    logger.info("Analysis agent processing...")
    
    try:
        llm = get_llm()
        chain = analysis_prompt | llm
        response = chain.invoke({
            "detection_result": state["detection_result"],
            "verdict": state["verdict"]
        })
        
        # Update state
        return {**state, "analysis": response.content}
    
    except Exception as e:
        logger.error(f"Error in analysis agent: {str(e)}")
        return {**state, "analysis": "Analysis could not be completed due to an error."}

def summary_agent(state: AgentState) -> AgentState:
    """Agent responsible for generating the final summary."""
    logger.info("Summary agent processing...")
    
    try:
        llm = get_llm()
        chain = summary_prompt | llm
        response = chain.invoke({
            "verdict": state["verdict"],
            "analysis": state["analysis"]
        })
        
        # Prepare final output
        final_output = {
            "verdict": state["verdict"],
            "analysis": state["analysis"],
            "summary": response.content
        }
        
        # Update state
        return {**state, "final_output": final_output}
    
    except Exception as e:
        logger.error(f"Error in summary agent: {str(e)}")
        return {**state, "final_output": {
            "verdict": state["verdict"],
            "analysis": state["analysis"],
            "summary": "Summary could not be generated due to an error."
        }}

# Define the agent workflow
def create_agent_workflow():
    """Create the multi-agent workflow using LangGraph."""
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("detector", detector_agent)
    workflow.add_node("analyzer", analysis_agent)
    workflow.add_node("summarizer", summary_agent)
    
    # Set the entry point
    workflow.set_entry_point("detector")

    # Define the edges (flow)
    workflow.add_edge("detector", "analyzer")
    workflow.add_edge("analyzer", "summarizer")
    workflow.add_edge("summarizer", END)
    
    # Compile the graph
    return workflow.compile()