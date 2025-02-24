import os
import streamlit as st
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_tool_calling_agent

import base64
import io
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
import seaborn as sns
import prophet
from datetime import datetime
import uuid
from datetime import datetime
from typing import TypedDict


# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["DEEPSEEK_API_KEY"] = st.secrets["general"]["DEEPSEEK_API_KEY"]

# Set LangGraph environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph data analysis - matplotlib"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

def save_plot(fig, filename=None):
    """Helper function to save plot to the images directory"""
    os.makedirs('images', exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f'plot_{timestamp}_{unique_id}.png'
    image_path = os.path.abspath(os.path.join('images', filename))
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return image_path

# Define state structure
class AgentState(TypedDict):
    df: pd.DataFrame
    user_query: str
    rephrased_query: str
    analysis_result: str
    image_path: str 

# Define nodes
def rephrase_query(state: AgentState) -> AgentState:
    """Node to rephrase user query into professional analysis question"""
    df_head = state["df"].head().to_string(index=False)

    prompt = f"""
    Original query: {state["user_query"]}

    Dataset Structure:
    - Columns: {', '.join(state["df"].columns)}
    - Sample Data:
    {df_head}

    Rephrase the original query into a professional data analysis question that:
    1. Specifies clear analysis objectives
    2. Considers appropriate statistical methods
    3. Identifies relevant columns for analysis
    4. Suggests potential visualization approaches
    """

    response = llm.invoke(prompt)
    return {**state, "rephrased_query": response.content.strip()}


def analyze_data(state: AgentState) -> AgentState:
    """Node to execute data analysis with Python REPL tool"""
    df = state["df"]
    rephrased_query = state["rephrased_query"]

    # Create Python REPL tool with required imports
    python_repl = PythonAstREPLTool(locals={
        "df": df,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "sklearn": sklearn,
        "prophet": prophet,
    })

    # Construct custom prompt with data context
    system_message = f"""
    You are a professional data analyst working with a pandas DataFrame.

    Dataset Context:
    - Shape: {df.shape[0]} rows, {df.shape[1]} columns
    - Columns: {', '.join(df.columns)}
    - Sample Data:
    {df.head(3).to_string(index=False)}

    Analysis Guidelines:
    1. Always validate data types and handle missing values first
    2. Convert data to appropriate types as needed before analysis
    3. Prefer vectorized pandas operations over loops
    4. For visualizations:
       - Create figure with fig = plt.figure(figsize=(10, 6))
       - Use matplotlib for basic plots
       - Use seaborn for complex plots
       - Include proper labels and titles
    5. Return intermediate results for verification
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Create agent with tool-calling capabilities
    agent = create_tool_calling_agent(llm, [python_repl], prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[python_repl],
        max_iterations = 15,
        verbose=True,
        handle_parsing_errors=True
    )

    # Enhanced analysis query
    analysis_prompt = f"""
    {rephrased_query}

    Execution Requirements:
    - Perform complete analysis including data cleaning

    - Data Cleaning Steps:
      1. First, print DataFrame column types
      2. Then, print missing values count
      3. Convert data to appropriate types as needed
        - Use pd.to_numeric() for numeric columns
        - Use pd.to_datetime() for date columns
      4. Ensure these commands are executed separately
      5. Use print() for each command
    
    - Data Analysis Guidelines:
      - Handle missing values appropriately
        - For numeric: fillna() with mean/median
        - For categorical: fillna() with mode
        - WARNING: due to changes in how Pandas handles inplace=True when using fillna(). Instead of using inplace=True, you should assign the result back to the column explicitly. 

    - Visualization Saving:
      - Generate only one visualization 
      - Save in high-quality format

    - Reporting:
      - Return final answer with markdown formatting
      - Provide brief explanation of each analysis step
    """

    try:
        plt.close('all')  # Clear previous figures
        result = agent_executor.invoke({"input": analysis_prompt})

        # Handle visualization output
        figs = [plt.figure(n) for n in plt.get_fignums()]
        image_path = ""
        if figs:
            fig = figs[0]
            image_path = save_plot(fig)
            plt.close(fig)
        else:  # Fallback visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            df.iloc[:, :3].hist(ax=ax)
            image_path = save_plot(fig)
            plt.close(fig)

        return {
            **state,
            "analysis_result": result.get("output", "No results generated"),
            "image_path": image_path
        }

    except Exception as e:
        plt.close('all')
        return {
            **state,
            "analysis_result": f"Analysis Error: {str(e)}",
            "image_path": ""
        }

# Build workflow
builder = StateGraph(AgentState)
builder.add_node("rephrase_query", rephrase_query)
builder.add_node("analyze_data", analyze_data)

# Define edges
builder.set_entry_point("rephrase_query")
builder.add_edge("rephrase_query", "analyze_data")
builder.add_edge("analyze_data", END)

# Compile agent
data_agent_visual = builder.compile()