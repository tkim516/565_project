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

st.set_page_config(page_title = "Home", 
                   page_icon = "📚")

st.title('Home')