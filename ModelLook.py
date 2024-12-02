import pandas as pd                  # Pandas
import numpy as np                   # Numpy
from matplotlib import pyplot as plt # Matplotlib

# Package to implement ML Algorithms
import sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest
import time
# Import MAPIE to calculate prediction intervals
from mapie.regression import MapieRegressor

# To calculate coverage score
from mapie.metrics import regression_coverage_score

# Package for data partitioning
from sklearn.model_selection import train_test_split

# Package to record time
import time

# Module to save and load Python objects to and from files
import pickle 

# Ignore Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')

import streamlit as st

st.set_page_config(page_title="My App", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")


st.markdown("<h1 style='text-align: center; color: black;'>Feature Importances 📈</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Dive deeper into the machine learning algorithms</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    model1 = st.selectbox('Choose your First Model',['Random Forest', 'XGBoost', 'Decision Tree', 'Ada Boost'])
    st.subheader("Model Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report", "ROC Curve"])

    with tab1:
        st.write("### Feature Importance")
        if model1 == 'Random Forest':
            st.image('RF_FI.png')
        elif model1 == 'XGBoost':
            st.image('XG_FI.png')
        elif model1 == 'Decision Tree':
            st.image('DT_FI.png')
        elif model1 == 'Ada Boost':
            st.image('ADA_FI.png')
        st.caption("Features used in this prediction are ranked by relative importance.")

    with tab2:
        st.write("### Confusion Matrix")
        if model1 == 'Random Forest':
            st.image('RF_CM.png')
        elif model1 == 'XGBoost':
            st.image('XG_CM.png')
        elif model1 == 'Decision Tree':
            st.image('DT_CM.png')
        elif model1 == 'Ada Boost':
            st.image('ADA_CM.png')
        st.caption("Confusion Matrix of model predictions.")

    with tab3:
        st.write("### Classification Report")
        if model1 == 'Random Forest':
            report_df = pd.read_csv('RF_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model1 == 'XGBoost':
            report_df = pd.read_csv('XG_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model1 == 'Decision Tree':
            report_df = pd.read_csv('DT_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model1 == 'Ada Boost':
            report_df = pd.read_csv('ADA_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))

    with tab4:
        st.write("### ROC Curve")
        if model1 == 'Random Forest':
            st.image('RF_ROC.png')
        elif model1 == 'XGBoost':
            st.image('XG_ROC.png')
        elif model1 == 'Decision Tree':
            st.image('DT_ROC.png')
        elif model1 == 'Ada Boost':
            st.image('ADA_ROC.png')

with col2:
    model2 = st.selectbox('Choose your Second Model',['Random Forest', 'XGBoost', 'Decision Tree', 'Ada Boost'])
    st.subheader("Model Insights")
    tab5, tab6, tab7, tab8 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report", "ROC Curve"])

    with tab5:
        st.write("### Feature Importance")
        if model2 == 'Random Forest':
            st.image('RF_FI.png')
        elif model2 == 'XGBoost':
            st.image('XG_FI.png')
        elif model2 == 'Decision Tree':
            st.image('DT_FI.png')
        elif model2 == 'Ada Boost':
            st.image('ADA_FI.png')
        st.caption("Features used in this prediction are ranked by relative importance.")

    with tab6:
        st.write("### Confusion Matrix")
        if model2 == 'Random Forest':
            st.image('RF_CM.png')
        elif model2 == 'XGBoost':
            st.image('XG_CM.png')
        elif model2 == 'Decision Tree':
            st.image('DT_CM.png')
        elif model2 == 'Ada Boost':
            st.image('ADA_CM.png')
        st.caption("Confusion Matrix of model predictions.")

    with tab7:
        st.write("### Classification Report")
        if model2 == 'Random Forest':
            report_df = pd.read_csv('RF_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model2 == 'XGBoost':
            report_df = pd.read_csv('XG_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model2 == 'Decision Tree':
            report_df = pd.read_csv('DT_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        elif model2 == 'Ada Boost':
            report_df = pd.read_csv('ADA_CR.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))


    with tab8:
        st.write("### ROC Curve")
        if model2 == 'Random Forest':
            st.image('RF_ROC.png')
        elif model2 == 'XGBoost':
            st.image('XG_ROC.png')
        elif model2 == 'Decision Tree':
            st.image('DT_ROC.png')
        elif model2 == 'Ada Boost':
            st.image('ADA_ROC.png')