import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Upload Data", 
                   page_icon="📚",
                   layout="wide")

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Upload your shopper data</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Unlock insights about shopper buying behaviors</h3>", unsafe_allow_html=True)


@st.cache_resource
def process_dataset(df, cat_vars):
    """
    One-hot encode categorical variables in the dataset.
    """
    df_encoded = pd.get_dummies(df, columns=cat_vars)
    return df_encoded


# Load models
with open('ada.pickle', 'rb') as f:
    ada_clf = pickle.load(f)

with open('dt.pickle', 'rb') as f:
    dt_clf = pickle.load(f)

with open('random_forest.pickle', 'rb') as f:
    rnd_forest_clf = pickle.load(f)

# Model selection
selected_model_str = st.selectbox('Choose Model', ['AdaBoost', 'Decision Tree', 'Random Forest'])
if selected_model_str == 'AdaBoost':
    model = ada_clf
elif selected_model_str == 'Decision Tree':
    model = dt_clf
elif selected_model_str == 'Random Forest':
    model = rnd_forest_clf

# File uploader and instructions
with st.expander("**How to prepare your data?** 📤"):
    st.write("""
    1. Export your data from your data source.
    2. Ensure your CSV file contains the columns mentioned in the sidebar.
    3. Upload the file here to proceed.
    """)

uploaded_file = st.file_uploader(
    "Upload your CSV file:",
    type=["csv"],
    help="Ensure the file is in CSV format with the required columns."
)

# Define categorical variables
cat_vars = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Month']

if uploaded_file:
    # Load uploaded data
    input_df = pd.read_csv(uploaded_file)

    # Ensure 'Revenue' is dropped if accidentally included
    if 'Revenue' in input_df.columns:
        input_df = input_df.drop(columns=['Revenue'])

    # Load training data for feature alignment
    train_df = pd.read_csv('online_shoppers_intention.csv')
    train_df_encoded = process_dataset(train_df, cat_vars)
    input_df_encoded = process_dataset(input_df, cat_vars)

    # Align features between train and input data
    feature_order = [col for col in train_df_encoded.columns if col != 'Revenue']  # Exclude target variable
    missing_columns = [col for col in feature_order if col not in input_df_encoded.columns]
    for col in missing_columns:
        input_df_encoded[col] = 0  # Add missing columns with default value

    # Reorder columns to match training data
    input_df_encoded = input_df_encoded[feature_order]

    # Make predictions
    predictions = model.predict(input_df_encoded)
    predictions_proba = model.predict_proba(input_df_encoded)

    # Display predictions
    st.success("File processed successfully!")
    st.write("Here are the predictions:")
    input_df['Predicted Revenue'] = predictions
    st.dataframe(input_df)

    st.write("Prediction Probabilities:")
    st.dataframe(pd.DataFrame(predictions_proba, columns=['Class 0 Probability', 'Class 1 Probability']))

else:
    st.info("Please upload a CSV file to start.")
# Define categorical variables
cat_vars = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Month']

# Load and process the training data
train_df = pd.read_csv('online_shoppers_intention.csv')
train_df_encoded = process_dataset(train_df, cat_vars)

# Define feature_order from the training data (exclude target variable 'Revenue')
feature_order = [col for col in train_df_encoded.columns if col != 'Revenue']

# Fix for manual input processing
with st.sidebar:
    st.markdown("### Manual Input")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    operating_systems = [1, 2, 3, 4, 5, 6, 7, 8]
    browsers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    regions = list(range(1, 10))  # Example range
    traffic_types = list(range(1, 21))  # Example range
    visitor_types = ['Returning_Visitor', 'New_Visitor']
    weekend = [True, False]

    # Input fields
    month = st.selectbox('Month', months)
    operating_system = st.selectbox('Operating System', operating_systems)
    browser = st.selectbox('Browser', browsers)
    region = st.selectbox('Region', regions)
    traffic_type = st.selectbox('Traffic Type', traffic_types)
    visitor_type = st.selectbox('Visitor Type', visitor_types)
    is_weekend = st.selectbox('Weekend', weekend)

    # Numeric inputs
    administrative = st.number_input('Administrative', min_value=0, value=0, step=1)
    administrative_duration = st.number_input('Administrative Duration', min_value=0.0, value=0.0)
    informational = st.number_input('Informational', min_value=0, value=0, step=1)
    informational_duration = st.number_input('Informational Duration', min_value=0.0, value=0.0)
    product_related = st.number_input('Product Related', min_value=0, value=0, step=1)
    product_related_duration = st.number_input('Product Related Duration', min_value=0.0, value=0.0)
    bounce_rates = st.number_input('Bounce Rates', min_value=0.0, value=0.0)
    exit_rates = st.number_input('Exit Rates', min_value=0.0, value=0.0)
    page_values = st.number_input('Page Values', min_value=0.0, value=0.0)
    special_day = st.number_input('Special Day', min_value=0.0, value=0.0)

    # Prepare manual input data
    manual_input = pd.DataFrame([{
        'Month': month,
        'OperatingSystems': operating_system,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type,
        'Weekend': is_weekend,
        'Administrative': administrative,
        'Administrative_Duration': administrative_duration,
        'Informational': informational,
        'Informational_Duration': informational_duration,
        'ProductRelated': product_related,
        'ProductRelated_Duration': product_related_duration,
        'BounceRates': bounce_rates,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_day
    }])

    # Process manual input
    manual_input_encoded = process_dataset(manual_input, cat_vars)
    missing_columns_manual = [col for col in feature_order if col not in manual_input_encoded.columns]
    for col in missing_columns_manual:
        manual_input_encoded[col] = 0
    manual_input_encoded = manual_input_encoded[feature_order]

    # Predict using manual input
    manual_prediction = model.predict(manual_input_encoded)
    manual_prediction_proba = model.predict_proba(manual_input_encoded)

st.subheader(f"Input form prediction: {manual_prediction[0]}")
max_probability = max(manual_prediction_proba[0])
# Display the maximum probability
st.subheader(f"Probability: {max_probability:.2f}")