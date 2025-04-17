import streamlit as st
from pages.regression import show_regression
from pages.clustering import show_clustering
from pages.neural_network import show_neural_network
from pages.llm import show_llm

def main():
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