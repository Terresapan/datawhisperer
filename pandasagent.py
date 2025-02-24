import os
import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek


# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Set LangGraph environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph data analysis"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define state structure
class AgentState(TypedDict):
    df: pd.DataFrame
    user_query: str
    rephrased_query: str
    analysis_result: str

# Define nodes

def summarize_data(state: AgentState) -> AgentState:
    df_head = state["df"].head().to_string(index=False)

    prompt = f"""
    Dataset columns: {', '.join(state["df"].columns)}

    First 5 rows of data:
    {df_head}

    You are a professional data scientist. 
    Please summarize the dataset and suggest five data analysis questions based on the dataset structure.
    Ensure the suggested questions are specific and actionable for data analysis.
    """

    summary = llm.invoke(prompt)
    return {**state, "summary": summary.content}

def rephrase_query(state: AgentState) -> AgentState:
    user_query = state["user_query"]

    # Get the first 5 rows of the dataset as a string
    df_head = state["df"].head().to_string(index=False)

    # Create a more informative prompt
    prompt = f"""
    User query: {user_query}

    Dataset columns: {', '.join(state["df"].columns)}

    First 5 rows of data:
    {df_head}

    You are a professional data scientist. You are helping a user analyze a dataset.
    Based on the user's query and the dataset structure, rephrase the query into a professional data analysis question.
    Ensure the rephrased question is specific and actionable for data analysis.
    """
    
    response = llm.invoke(prompt)
    return {**state, "rephrased_query": response.content}

   
def analyze_data(state: AgentState) -> AgentState:
    agent = create_pandas_dataframe_agent(
        llm,
        state["df"],
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=True  # Set to True for debugging
    )

    query = f"""

    {state["rephrased_query"]}

    - Important: You have access to the **entire dataset**, not just a sample. 
    - Ensure you process all rows in the dataset for analysis. 
    - Always validate data types, convert to appropriate types as needed, and handle missing values first
    - You should always execute the code and print out your analysis results to answer the question.
    """
    assert state["df"].shape[0] > 5, "The dataset must have more than 5 rows for analysis."

    # Invoke the agent with the query
    result = agent.invoke(query)
    analysis_output = result.get("output", "No analysis output generated.")  # Safer handling

    return {**state, "analysis_result": result["output"]}

# Build workflow
builder = StateGraph(AgentState)
builder.add_node("rephrase_query", rephrase_query)
builder.add_node("analyze_data", analyze_data)

# Define edges
builder.set_entry_point("rephrase_query")
builder.add_edge("rephrase_query", "analyze_data")
builder.add_edge("analyze_data", END)

# Compile with checkpointing
data_agent = builder.compile()

