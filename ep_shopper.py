import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Title and description of the app
st.title('Online Shopper Conversion: A Machine Learning App') 
# Display gif
# st.image('fetal_health_image.gif', width = 600)

st.write("Utalize the advanced machine learning application to predict if an online shopper will convert.")

# Random Forest:
rf_pickle = open('random_forest_fetal_health.pickle', 'rb') 
model_cv = pickle.load(rf_pickle) 
rf_pickle.close()

# Load the default dataset
default_df = pd.read_csv('online_shoppers_intention.csv')

##### -------------------   SIDEBAR   ---------------------- ######
# Create a sidebar for input collection

## st.sidebar.image('traffic_sidebar.jpg', use_column_width = True, caption = "Shopper Conversion Predictor")
st.sidebar.header('Online Shopper Features Input')

st.sidebar.write('You can either upload your data file or manually enter input features.')

# Option 1: Asking users to input their data as a file
with st.sidebar.expander('Option 1: Upload CSV file', expanded=False):
    file = st.file_uploader('Upload your CSV file', type=["csv"])
    st.write("Sample Data Format for Upload")
    st.write(default_df.head(5))
    st.warning("Ensure your uploaded file has the same column names and data types as shown above.")


# Option 2: Asking users to input their data using a form in the sidebar
with st.sidebar.expander('Option 2: Fill out form', expanded=False):
    st.header("Enter Your Details")
    with st.form('user_inputs_form'):

        # Categorical Columns
        # Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue
        month = st.selectbox('Choose a month', options=default_df['Month'].unique())
        operating_system = st.selectbox('Choose an operating system', options=default_df['OperatingSystems'].unique())
        browser = st.selectbox('Choose a browser', options=default_df['Browser'].unique())
        region = st.selectbox('Choose a region', options=default_df['Region'].unique())
        traffic_type = st.selectbox('Choose a traffic type', options=default_df['TrafficType'].unique())
        visitor_type = st.selectbox('Choose visitor type', options=default_df['VisitorType'].unique())
        weekend = st.selectbox('Is it a weekend?', options=default_df['Weekend'].unique())
        revenue = st.selectbox('Did the visit result in revenue?', options=default_df['Revenue'].unique())

        # Numerical Columns
        # Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay
        administrative = st.slider('Number of administrative pages visited', min_value=int(default_df['Administrative'].min()), max_value=int(default_df['Administrative'].max()), step=1)
        administrative_duration = st.slider('Total duration spent on administrative pages', min_value=default_df['Administrative_Duration'].min(), max_value=default_df['Administrative_Duration'].max(), step=0.1)
        informational = st.slider('Number of informational pages visited', min_value=int(default_df['Informational'].min()), max_value=int(default_df['Informational'].max()), step=1)
        informational_duration = st.slider('Total duration spent on informational pages', min_value=default_df['Informational_Duration'].min(), max_value=default_df['Informational_Duration'].max(), step=0.1)
        product_related = st.slider('Number of product-related pages visited', min_value=int(default_df['ProductRelated'].min()), max_value=int(default_df['ProductRelated'].max()), step=1)
        product_related_duration = st.slider('Total duration spent on product-related pages', min_value=default_df['ProductRelated_Duration'].min(), max_value=default_df['ProductRelated_Duration'].max(), step=0.1)
        bounce_rates = st.slider('Bounce rates', min_value=default_df['BounceRates'].min(), max_value=default_df['BounceRates'].max(), step=0.01)
        exit_rates = st.slider('Exit rates', min_value=default_df['ExitRates'].min(), max_value=default_df['ExitRates'].max(), step=0.01)
        page_values = st.slider('Page values', min_value=default_df['PageValues'].min(), max_value=default_df['PageValues'].max(), step=0.1)
        special_day = st.slider('Closeness to a special day', min_value=default_df['SpecialDay'].min(), max_value=default_df['SpecialDay'].max(), step=0.1)
    
        # Submit Form Button
        submit_button = st.form_submit_button("Submit Form Data")


if file is None:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['Revenue'])

    # Combine the list of user data as a row to default_df
    #encode_df.loc[len(encode_df)] = [month,operating_system,browser,region,traffic_type,visitor_type,weekend,revenue,administrative,administrative_duration,informational,informational_duration,product_related,product_related_duration,bounce_rates,exit_rates,page_values,special_day]


    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [administrative,administrative_duration,informational,informational_duration,product_related,product_related_duration,bounce_rates,exit_rates,page_values,special_day,month,operating_system,browser,region,traffic_type,visitor_type,weekend]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    st.write("")

        # Get the prediction with its intervals
    alpha = st.slider("Select alpha value for prediction intervals", 
                        min_value=0.01, 
                        max_value=0.5, 
                        step=0.01)

    prediction, intervals = model_cv.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[0, :]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Hourly I-94 ATR 301 reported westbound traffic volume", value = f"{pred_value :.0f}")
    st.write(f"With a {(1 - alpha)* 100:.0f}% confidence interval:")
    st.write(f"**Confidence Interval**: [{lower_limit:.2f}, {upper_limit:.2f}]")