import streamlit as st
import os
import platform

# Disable Streamlit's file watcher to avoid PyTorch conflicts
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Import asyncio and nest_asyncio after setting environment variables
import asyncio
import nest_asyncio

# Configure asyncio for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Import pages after asyncio setup
from pages.regression import show_regression
from pages.clustering import show_clustering
from pages.neural_network import show_neural_network
from pages.llm import show_llm

def main():
    # Configure Streamlit to handle async operations
    if not hasattr(st.session_state, 'async_loop'):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        st.session_state.async_loop = loop

    st.set_page_config(
        page_title="AI Analysis Dashboard",
        page_icon="🤖",
        layout="wide"
    )

    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Use radio buttons for navigation
    page = st.sidebar.radio(
        "Go to",
        ["Regression", "Clustering", "Neural Network", "Large Language Model"],
        index=0  # Default to first page
    )

    # Display the selected page
    if page == "Regression":
        show_regression()
    elif page == "Clustering":
        show_clustering()
    elif page == "Neural Network":
        show_neural_network()
    elif page == "Large Language Model":
        show_llm()

if __name__ == "__main__":
    main() 