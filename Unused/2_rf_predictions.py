import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title = "Predictions", 
                   page_icon = "📚")

# Page title
st.markdown(
    """
    <h2 style="text-align: center; color: #4a4a4a;">Online Shopper Conversion: Random Forest Results 🌟</h2>
    """,
    unsafe_allow_html=True,
)

# Subtitle with a descriptive message
st.markdown(
    """
    <h3 style="text-align: center; color: #2b50aa;">Discover Your Conversion Chances</h3>
    <p style="text-align: center; color: #6c757d; font-size: 1.1rem;">
    Utalize the advanced machine learning application to predict if an online shopper will convert.
    </p>
    <hr style="border: 1px solid #ccc;">
    """,
    unsafe_allow_html=True,
)

# Ensure inputs are available
if 'gre_score' not in st.session_state:
    st.warning("Please fill out the inputs on the 'User Input' page first!")

# Load the trained model and dataset
with open('random_forest_fetal_health.pickle', 'rb') as rf_model_file:
    model_cv = pickle.load(rf_model_file)

default_df = pd.read_csv('online_shoppers_intention.csv')


########### WIP ################

# Prepare user input data
user_data = {
    'Number of administrative pages visited': st.session_state['administrative'],
    'Total duration spent on administrative pages': st.session_state['administrative_duration'],
    'Number of informational pages visited': st.session_state['informational'],
    'Total duration spent on informational pages': st.session_state['informational_duration'],
    'Number of product-related pages visited': st.session_state['product_related'],
    'Total duration spent on product-related pages': st.session_state['product_related_duration'],
    'Bounce rates': st.session_state['bounce_rates'],
    'Exit rates': st.session_state['exit_rates'],
    'Page values': st.session_state['page_values'],
    'Closeness to a special day': st.session_state['special_day']
}
user_df = pd.DataFrame([user_data])

# Combine user data with default dataset for dummy encoding
encode_df = default_df.copy().drop(columns=['Revenue'])

# Ensure the order of columns in user data is in the same order as that of original data
user_df = user_df[encode_df.columns]

# Concatenate two dataframes together along rows (axis = 0)
encode_df = pd.concat([encode_df, user_df], axis = 0)
encode_dummy_df = pd.get_dummies(encode_df).tail(1)

# Model prediction with confidence intervals
alpha = 0.1  # 90% confidence level
prediction, intervals = model_cv.predict(encode_dummy_df, alpha=alpha)
pred_value = prediction[0]
lower_limit = max(0, intervals[:, 0][0][0])
upper_limit = min(1, intervals[:, 1][0][0])

# Display prediction results
st.metric(label="Predicted Conversion Probability", value=f"{pred_value * 100:.2f}%")
st.write("With a 90% confidence interval:")
st.write(f"**Confidence Interval**: [{lower_limit * 100:.2f}%, {upper_limit * 100:.2f}%]")

# Tabs for visualizations
st.markdown("<h3>Model Insights</h3>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])
with tab1:
    st.image('feature_imp.svg', caption="Relative importance of features in prediction.")
with tab2:
    st.image('residual_plot.svg', caption="Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.image('pred_vs_actual.svg', caption="Visual comparison of predicted and actual values.")
with tab4:
    st.image('coverage.svg', caption="Range of predictions with confidence intervals.")