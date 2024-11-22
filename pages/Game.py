# Streamlit webapp

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Interactive game where students try to beat the model
# Select model
# Show dataset imbalance
# Show feature importance for model
# Show row of data
# Allow players to guess if a customer will buy or not
# Provide prediction from model and correct answer
# Go to next question

st.set_page_config(page_title = "Game", 
                   page_icon = "📚",
                   layout= "wide")

st.title('Can you beat a machine learning model?')

@st.cache_resource
def load_and_cache_csv(file_name):
    return pd.read_csv(file_name)

@st.cache_resource
def load_and_cache_pickle(file_name):
    with open(file_name, 'rb') as model_pickle:
        return pickle.load(model_pickle)

# Load dataset and models
df = load_and_cache_csv('online_shoppers_intention.csv')
ada_clf = load_and_cache_pickle('ada.pickle')

dataset_imbalance_series = df['Revenue'].value_counts(normalize=True)

dataset_imbalance_df = pd.DataFrame({
    'Revenue': ['True', 'False'], 
    'Proportion': [dataset_imbalance_series[0], dataset_imbalance_series[1]]})

import pandas as pd

# Define categorical variables for one-hot encoding
cat_vars = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Month']

# Load datasets
question_df = pd.read_csv('question_dataset.csv')
train_df = pd.read_csv('online_shoppers_intention.csv')

# One-hot encode categorical variables
train_df_encoded = pd.get_dummies(train_df, columns=cat_vars)
question_df_encoded = pd.get_dummies(question_df, columns=cat_vars)

# Align features between train_df_encoded and question_df_encoded
feature_order = train_df_encoded.columns

# Add missing columns to question_df_encoded
missing_columns = [col for col in feature_order if col not in question_df_encoded.columns]
for col in missing_columns:
    question_df_encoded[col] = False

# Reorder columns to match training data
question_df_encoded = question_df_encoded[feature_order]

# Split training data into X and y
if 'Revenue' in train_df_encoded.columns:
    X = train_df_encoded.drop(columns='Revenue')
    y = train_df_encoded['Revenue']
else:
    raise ValueError("The 'Revenue' column is missing in the train dataset.")



# Custom CSS to modify sidebar width and font size
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
        [data-testid="stSidebar"] * {
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar content
with st.sidebar:
    st.title('Column Descriptions')
    
    st.header('PageValues ($)')
    st.write('Average monetary contribution of a page, calculated by dividing the total revenue generated from sessions that included the page by the number of sessions involving the page')
    
    st.header('ProductRelated_Duration (Seconds)')
    st.write('Total time spent on product-related pages')

    st.header('BounceRates (%)')
    st.write('Percentage of visitors who enter and leave the site without further interaction')

    st.header('ExitRates (%)')
    st.write('Percentage of pageviews that were the last in the session')
    
    st.header('Administrative_Duration (Seconds)')
    st.write('Total time spent on administrative pages ')
    
    st.header('ProductRelated (Count)')
    st.write('Number of product-related pages visited during the session')


# Ideal solution
# Game screen with username and instructions
# Username box hidden question 1 by 1 with feedback until 5 questions reached
total_questions = 7

if "username" not in st.session_state:
  st.session_state.user_results = {}
  st.session_state.model_results = {}

  with st.form(key="username_form"):
    username_input = st.text_input("Enter your username")
    username_submit_button = st.form_submit_button(label="Enter Game")
    
    st.write('Choose your opponent')
    st.image('adaboost_avatar.webp')
    # Save the username to session state upon form submission
    if username_submit_button:
        if username_input.strip():
                st.session_state.username = username_input.strip()
                st.success(f"Welcome, {st.session_state.username}!")
                time.sleep(1)  # Brief pause to show the success message
                st.rerun()  # Re-run the app to update the state
        else:
            st.error("Username cannot be empty. Please try again.") 
       
elif "username" in st.session_state:
  #user_results_dict['username'] = st.session_state.username
  #st.session_state.user_results = user_results_dict
  st.session_state.user_results['username'] = st.session_state.username

  if "question_num" not in st.session_state:
      st.session_state.question_num = 0
  if "quiz_complete" not in st.session_state:
      st.session_state.quiz_complete = False

  def get_new_question_return_answer(question_df, question_num):
      row = question_df.iloc[[question_num]]  # Use question_num to index directly
      correct_answer = row['Revenue'].iloc[0]

      st.header(f'Question {question_num + 1} of {total_questions}')
      st.dataframe(row[[
          'PageValues',
          'ProductRelated_Duration',
          'BounceRates',
          'ExitRates',
          'Administrative_Duration',
          'ProductRelated']], 
          hide_index=True)
      
      row_encoded = X.iloc[[question_num]]
      model_predicton = ada_clf.predict(row_encoded)
            
      return correct_answer, model_predicton

  # Show the current question
  if not st.session_state.quiz_complete:
      question_num = st.session_state.question_num
      correct_answer, model_prediction = get_new_question_return_answer(question_df, question_num)

      with st.form(key=f'user_prediction_form_{question_num}'):
          user_prediction = st.selectbox(
              label='Predict if the customer will buy a product',
              options=[True, False]
          )
          submit_button = st.form_submit_button(label='Submit')
      if submit_button:
            if user_prediction == correct_answer:
                st.session_state.user_results[question_num] = True
                #user_results_dict[question_num] = True
                st.success('Correct!')

            if user_prediction != correct_answer:
                st.session_state.user_results[question_num] = False
              #user_results[question_num] = False
                st.error(f'Incorrect. The correct answer is {correct_answer}')

            if model_prediction == correct_answer:
                st.session_state.model_results[question_num] = True
                st.header('MODEL SUCCESS')

            if model_prediction != correct_answer:
                st.session_state.model_results[question_num] = False
                st.header('FAIL')

          # Increment the question number
            st.session_state.question_num += 1    
          #st.session_state.user_results = user_results_dict
      
          # Check if the quiz is complete
            if st.session_state.question_num == total_questions:
                st.session_state.quiz_complete = True
            
            time.sleep(1)
            st.rerun()

  # Show the completion message
  if st.session_state.quiz_complete:
        user_num_correct = 0
        model_num_correct = 0
        for i in range(total_questions):
            user_question_result = st.session_state.user_results[i]
            if user_question_result == True:
                user_num_correct += 1
            
            model_question_result = st.session_state.model_results[i]
            if model_question_result == True:
                model_num_correct += 1

        if user_num_correct > model_num_correct:
            st.balloons()
            st.header("Great job, you beat the model!")
            
        elif user_num_correct < model_num_correct:
            st.header("Womp womp, you failed to beat the model.")

        st.subheader(f'Your score: {user_num_correct} out of {total_questions}')
        st.write(st.session_state.user_results)
        st.subheader(f'Model score: {model_num_correct} out of {total_questions}')
        st.write(st.session_state.model_results)


