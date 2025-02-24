import os
import uuid
import streamlit as st
import pandas as pd
from pandasagent import data_agent, summarize_data
from visualdata import data_agent_visual
from prebuilt import show_prebuilt_analysis
from utils import save_feedback, check_password
from pygwalker.api.streamlit import StreamlitRenderer
import matplotlib.pyplot as plt

# Page config for wider layout
st.set_page_config(layout="wide", page_title="Data Whisperer", page_icon="📊")

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Setup the sidebar with instructions and feedback form."""
    st.sidebar.header("📈 Data Whisperer")
    st.sidebar.markdown(
        "This app translates complex datasets into simple, strategic insights. "
        "It helps you visualise your data and make informed decisions."
    )
    
    st.sidebar.write("### Instructions")
    st.sidebar.write(
        "1. :key: Enter password to access the app\n"
        "2. :pencil: Upload your data file (CSV files only)\n"
        "3. :mag: Select files to analyze\n"
        "4. :speech_balloon: Chat with your data, ask questions, and visualize insights\n"
    )

    # Feedback section
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("💭 Feedback")
    feedback = st.sidebar.text_area(
        "Share your thoughts",
        value=st.session_state.feedback,
        placeholder="Your feedback helps us improve..."
    )

    if st.sidebar.button("📤 Submit Feedback"):
        if feedback:
            try:
                save_feedback(feedback)
                st.session_state.feedback = ""
                st.sidebar.success("✨ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error(f"❌ Error saving feedback: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter feedback before submitting")

    st.sidebar.image("assets/agentgarden.jpg", use_container_width=True)

# Main Streamlit app
def main():
    """Main application function."""
    setup_sidebar()
    
    if not check_password():
        st.stop()

    # Session state setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "visualization_history" not in st.session_state:
        st.session_state.visualization_history = []

    # File upload section with styling
    with st.container():
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            if st.session_state.df is None:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")

            # Add a "Summarize Data" button
            if st.button("📊 Summarize Data", type="secondary"):
                st.session_state.current_tab = "📊 Summarize Data"
                with st.spinner("🔍 Summarizing your data..."):
                    temp_state = {
                        "df": st.session_state.df,
                        "user_query": "",
                        "rephrased_query": "",
                        "analysis_result": "",
                    }
                    result = summarize_data(temp_state)
                    st.session_state.summary = result["summary"]
                    st.rerun()  # Rerun the app to update the tab

    # Create styled navigation
    current_tab = st.session_state.get("current_tab", "📊 Summarize Data")
    current_tab = st.radio(
        "Select View:",
        ["📊 Summarize Data", "🔍 Analyze Data", "📈 Visualize Data", "👀 Prebuilt Analysis", "⚡ Work with Data"],
        horizontal=True,
        label_visibility="hidden",
        index=["📊 Summarize Data", "🔍 Analyze Data", "📈 Visualize Data", "👀 Prebuilt Analysis", "⚡ Work with Data"].index(current_tab)
    )

    st.markdown("---")

    # Tab content
    if current_tab == "📊 Summarize Data":
        if st.session_state.df is not None:
            st.subheader("📝 Data Summary")
            if st.session_state.summary:
                st.write(st.session_state.summary)
            else:
                st.info("ℹ️ Click the 'Summarize Data' button to generate a summary.")
            
            st.subheader("👀 Sample Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        else:
            st.info("👆 Please upload a CSV file to view data summary")

    elif current_tab == "🔍 Analyze Data":
        if st.session_state.df is not None:
            # Display existing analysis history
            for query, result in st.session_state.analysis_history:
                st.markdown("### ❓ Question")
                st.markdown(query)
                st.markdown("### 💡 Answer")
                st.markdown(result)
                st.markdown("---")

            user_input = st.chat_input("💭 Ask me anything about the data!")
            
            if user_input:
                with st.spinner("🤔 Analyzing your data..."):
                    state = {
                        "df": st.session_state.df,
                        "user_query": user_input,
                        "rephrased_query": "",
                        "analysis_result": "",
                    }
                    
                    response = data_agent.invoke(state)
                    
                    # Store the new Q&A pair in history
                    st.session_state.analysis_history.append((user_input, response["analysis_result"]))
                    
                    # Display the new result
                    st.markdown("### ❓ Question")
                    st.markdown(user_input)
                    st.markdown("### 💡 Answer")
                    st.markdown(response["analysis_result"])
                    st.markdown("---")
        else:
            st.info("👆 Please upload a CSV file to analyze data")

    # Visualization section
    elif current_tab == "📈 Visualize Data":
        if st.session_state.df is not None:
            # Display existing visualization history
            for query, result, image_path in st.session_state.visualization_history:
                st.markdown("### ❓ Question")
                st.markdown(query)
                st.markdown("### 💡 Answer")
                st.markdown(result)
                if image_path and os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
                else:
                    st.warning("Visualization not available")
                st.markdown("---")

            user_input = st.chat_input("💭 Ask me to visualize the data!")
            
            if user_input:
                with st.spinner("🤔 Visualizing your data..."):
                    plt.close('all')  # Cleanup before analysis
                    try:
                        state = {
                            "df": st.session_state.df,
                            "user_query": user_input,
                            "rephrased_query": "",
                            "analysis_result": "",
                            "image_path": ""
                        }
                        
                        # Invoke the agent
                        response = data_agent_visual.invoke(state)
                        
                        # Store results in history
                        st.session_state.visualization_history.append(
                            (user_input, response['rephrased_query'], response["analysis_result"], response["image_path"])
                        )
                        
                        # Display results
                        st.markdown("### ❓ Question")
                        st.markdown(user_input)
                        st.markdown("### 🔍 Data Analysis Plan")
                        st.markdown(response["rephrased_query"])
                        st.markdown("### 💡 Answer")
                        st.markdown(response["analysis_result"])
                        
                        if response["image_path"] and os.path.exists(response["image_path"]):
                            st.image(response["image_path"], use_container_width=True)
                        else:
                            st.warning("No visualization generated")
                        
                        st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Visualization Failed: {str(e)}")
                        st.session_state.visualization_history.append(
                            (user_input, f"Error: {str(e)}", "")
                        )
                    
                    finally:
                        plt.close('all')  # Ensure cleanup
        else:
            st.info("👆 Please upload a CSV file to visualize data")

    elif current_tab == "👀 Prebuilt Analysis":
        if st.session_state.df is not None:
            show_prebuilt_analysis()
        else:
            st.info("👆 Please upload a CSV file to analyze data")

    else:  # Work with Data tab
        if st.session_state.df is not None:
            df = pd.read_csv(uploaded_file)
            pyg_app = StreamlitRenderer(df)
            pyg_app.explorer()
        else:
            st.info("👆 Please upload a CSV file to work with data")

if __name__ == "__main__":
    main()