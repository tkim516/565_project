# Streamlit webapp

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Interactive game where students try to beat the model
# Select model
# Show dataset imbalance
# Show feature importance for model
# Show row of data
# Allow players to guess if a customer will buy or not
# Provide prediction from model and correct answer
# Go to next question

#########################
# Setting the page configuration
st.set_page_config(
    page_title="Home",
    page_icon="🎮",
    layout="centered"
)

# Centered Title using HTML and Markdown
st.markdown(
    """
    <h2 style="text-align: center; color: #69503c;">Interactive Model Challenge: Can You Beat the Model? 🎯</h2>
    """,
    unsafe_allow_html=True,
)

# Subtitle with styling
st.markdown(
    """
    <h3 style="text-align: center; color: #1c2d8f; font-family: Arial, sans-serif;">
    Test your intuition against machine learning models and learn about data science along the way!
    </h3>
    """,
    unsafe_allow_html=True,
)

# Insert an image (optional)
st.image("interactive_game.jpg", caption="Compete with AI and enhance your data science skills!", use_column_width=True)

# Sidebar navigation header
with st.sidebar:
    st.header("🔍 Navigate the App")
    st.write("Use the links below to explore different sections:")
    st.markdown("""
    - **Home**: Welcome page
    - **User Form**: Upload your CSV file.
    - **Dataset Imbalance**: Understand the distribution of the dataset
    - **Feature Importance**: Learn about key features in the model
    - **Game**: Play the interactive game and guess outcomes
    """)

st.sidebar.info("Select a section from the sidebar to get started.")


#### Needs to be Updated #######

# Interactive introduction with expander
with st.expander("**What can you do with this app?**"):
    st.write("""
    🎮 **Select Model**: Choose from various machine learning models to compete against.
    
    📊 **Understand Dataset Imbalance**: Explore the distribution of classes in the dataset.
    
    🔑 **Feature Importance**: Discover which features are most influential in the model's predictions.
    
    🎲 **Interactive Game**: Guess whether a customer will buy or not based on the data. Compare your intuition with the model's prediction.

    """)

############

# Add a footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #777; font-size: 14px; font-family: Arial, sans-serif;">
    Made with ❤️ by your Machine Learning team.
    </p>
    """,
    unsafe_allow_html=True,
)
