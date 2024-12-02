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

st.title("Dataset Imbalance :bar_chart:")
st.subheader("Explore the imbalance in the dataset's responses")

dfi = pd.read_csv('online_shoppers_intention.csv')
dfi_filF = dfi[dfi['Revenue'] == False]
dfi_filF.reset_index(drop=True, inplace=True)
dfi_filT = dfi[dfi['Revenue'] == True]
dfi_filT.reset_index(drop=True, inplace=True)

FAdministrative = str(round(dfi_filF['Administrative'].mean(),2))
FAdministrative_D = str(round(dfi_filF['Administrative_Duration'].mean(),2))
FInformational = str(round(dfi_filF['Informational'].mean(),2))
FInformational_D = str(round(dfi_filF['Informational_Duration'].mean(),2))
FProductRelated = str(round(dfi_filF['ProductRelated'].mean(),2))
FProductRelated_D = str(round(dfi_filF['ProductRelated_Duration'].mean(),2))
FBounceR = str(round(dfi_filF['BounceRates'].mean(),2))
FExitR = str(round(dfi_filF['ExitRates'].mean(),2))
FPageV = str(round(dfi_filF['PageValues'].mean(),2))
FSpecialD = str(round(dfi_filF['SpecialDay'].mean(),2))
FMonth = str(dfi_filF['Month'].mode()[0])
FOperatingS = str(dfi_filF['OperatingSystems'].mode()[0])
FBrowser = str(dfi_filF['Browser'].mode()[0])
FRegion = str(dfi_filF['Region'].mode()[0])
FTrafficT = str(dfi_filF['TrafficType'].mode()[0])
FVisitorT = str(dfi_filF['VisitorType'].mode()[0])
if dfi_filF['Weekend'].mode()[0] == False:
    FWeekend = 'Yes'
elif dfi_filF['Weekend'].mode()[0] == True:
    FWeekend = 'No'

TAdministrative = str(round(dfi_filT['Administrative'].mean(),2))
TAdministrative_D = str(round(dfi_filT['Administrative_Duration'].mean(),2))
TInformational = str(round(dfi_filT['Informational'].mean(),2))
TInformational_D = str(round(dfi_filT['Informational_Duration'].mean(),2))
TProductRelated = str(round(dfi_filT['ProductRelated'].mean(),2))
TProductRelated_D = str(round(dfi_filT['ProductRelated_Duration'].mean(),2))
TBounceR = str(round(dfi_filT['BounceRates'].mean(),2))
TExitR = str(round(dfi_filT['ExitRates'].mean(),2))
TPageV = str(round(dfi_filT['PageValues'].mean(),2))
TSpecialD = str(round(dfi_filT['SpecialDay'].mean(),2))
TMonth = str(dfi_filT['Month'].mode()[0])
TOperatingS = str(dfi_filT['OperatingSystems'].mode()[0])
TBrowser = str(dfi_filT['Browser'].mode()[0])
TRegion = str(dfi_filT['Region'].mode()[0])
TTrafficT = str(dfi_filT['TrafficType'].mode()[0])
TVisitorT = str(dfi_filT['VisitorType'].mode()[0])
if dfi_filT['Weekend'].mode()[0] == False:
    TWeekend = 'Yes'
elif dfi_filT['Weekend'].mode()[0] == True:
    TWeekend = 'No'

Filt = st.selectbox('Would you like to see confirmed sales or missed sales?', ['Confirmed Sales', 'Missed Sales'])
F_count = str(dfi_filF.shape[0])
T_count = str(dfi_filT.shape[0])

if Filt == 'Confirmed Sales':
    st.write(dfi_filT)
    st.subheader("This dataset has " + T_count + " rows in it")
    st.markdown("<h5 style='text-align: left; color: black;'>Below are the averages for quantitative variables and modes for categorical variables</h5Ss>", unsafe_allow_html=True)
    st.write(f'Administrative: **{TAdministrative}**')
    st.write(f'Administrative Duration: **{TAdministrative_D}**')
    st.write(f'Informational: **{TInformational}**')
    st.write(f'Informational Duration: **{TInformational_D}**')
    st.write(f'Product Related: **{TProductRelated}**')
    st.write(f'Product Related Duration: **{TProductRelated_D}**')
    st.write(f'Bounce Rates: **{TBounceR}**')
    st.write(f'Exit Rates: **{TExitR}**')
    st.write(f'Page Values: **{TPageV}**')
    st.write(f'Special Day: **{TSpecialD}**')
    st.write(f'Month: **{TMonth}**')
    st.write(f'Operating Systems: **{TOperatingS}**')
    st.write(f'Browser: **{TBrowser}**')
    st.write(f'Region: **{TRegion}**')
    st.write(f'Traffic Type: **{TTrafficT}**')
    st.write(f'Visitor Type: **{TVisitorT}**')
    st.write(f'Weekend: **{TWeekend}**')
    
elif Filt == 'Missed Sales':
    st.write(dfi_filF)
    st.subheader("This dataset has " + F_count + " rows in it")
    st.markdown("<h5 style='text-align: left; color: black;'>Below are the averages for quantitative variables and modes for categorical variables</h5Ss>", unsafe_allow_html=True)
    st.write(f'Administrative: **{FAdministrative}**')
    st.write(f'Administrative Duration: **{FAdministrative_D}**')
    st.write(f'Informational: **{FInformational}**')
    st.write(f'Informational Duration: **{FInformational_D}**')
    st.write(f'Product Related: **{FProductRelated}**')
    st.write(f'Product Related Duration: **{FProductRelated_D}**')
    st.write(f'Bounce Rates: **{FBounceR}**')
    st.write(f'Exit Rates: **{FExitR}**')
    st.write(f'Page Values: **{FPageV}**')
    st.write(f'Special Day: **{FSpecialD}**')
    st.write(f'Month: **{FMonth}**')
    st.write(f'Operating Systems: **{FOperatingS}**')
    st.write(f'Browser: **{FBrowser}**')
    st.write(f'Region: **{FRegion}**')
    st.write(f'Traffic Type: **{FTrafficT}**')
    st.write(f'Visitor Type: **{FVisitorT}**')
    st.write(f'Weekend: **{FWeekend}**')
else:
    pass
