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
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Explore different machine learning models</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Visualize and compare performance</h3>", unsafe_allow_html=True)


# Insert an image (optional)
st.image("ecommerce.png", use_column_width=True)

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
    
    🎲 **Interactive Game**: Predict whether a customer will buy or not based on the data. Compare your intuition with the model's prediction.

    """)

############

# Add a footer
st.markdown(
    """
    <p style="text-align: center; color: #777; font-size: 14px; font-family: Arial, sans-serif;">
    By Tyler, Shivam, Michael, and Elan.
    </p>
    """,
    unsafe_allow_html=True,
)
