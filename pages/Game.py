# Streamlit webapp

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import timedelta

st.set_page_config(page_title = "Game", 
                   page_icon = "📚",
                   layout= "wide")
#if 'tutorial_complete' in st.session_state and st.session_state.tutorial_complete == False:
#   st.title('Tutorial')

st.title('Predict if a customer will make a purchase')

@st.cache_resource
def load_and_cache_csv(file_name):
    return pd.read_csv(file_name)

@st.cache_resource
def load_and_cache_pickle(file_name):
    with open(file_name, 'rb') as model_pickle:
        return pickle.load(model_pickle)

@st.cache_resource
def process_dataset(df, cat_vars):
    df_encoded = pd.get_dummies(df, columns=cat_vars)
    return df_encoded

# Load dataset and models
df = load_and_cache_csv('online_shoppers_intention.csv')
ada_clf = load_and_cache_pickle('ada.pickle')
dt_clf = load_and_cache_pickle('dt.pickle')
rnd_forest_clf = load_and_cache_pickle('random_forest.pickle')

dataset_imbalance_series = df['Revenue'].value_counts(normalize=True)

dataset_imbalance_df = pd.DataFrame({
    'Revenue': ['True', 'False'], 
    'Proportion': [dataset_imbalance_series[0], dataset_imbalance_series[1]]})

# Define categorical variables for one-hot encoding
cat_vars = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Month']

# Load datasets
question_df = pd.read_csv('question_dataset.csv')
train_df = pd.read_csv('online_shoppers_intention.csv')

# One-hot encode categorical variables

train_df_encoded = process_dataset(train_df, cat_vars)
question_df_encoded = process_dataset(question_df, cat_vars)

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
          hide_index=True,
          width=1500)
      
        row_encoded = X.iloc[[question_num]]
        ada_predicton = ada_clf.predict(row_encoded)
        dt_prediction = dt_clf.predict(row_encoded)
        rnd_forest_prediction = rnd_forest_clf.predict(row_encoded)

        model_predictions_dict = {'AdaBoost': ada_predicton, 'Decision Tree': dt_prediction, 'Random Forest': rnd_forest_prediction }
    
        return correct_answer, model_predictions_dict


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

  st.markdown(
    """
    <style>
    .big-font {
        font-size:24px !important;
        margin-bottom: -30px;
    }
    </style>
    """, 
    unsafe_allow_html=True)

  with st.form(key="username_form"):
    st.markdown('<p class="big-font">Enter your username</p>', unsafe_allow_html=True)
    username_input = st.text_input("", key="username_input")
    username_submit_button = st.form_submit_button(label="Submit")
    
    #st.image('.webp')
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
  if 'user_start_time' not in st.session_state:
    st.session_state.user_start_time = None
  if 'tutorial_complete' not in st.session_state:
    st.session_state.tutorial_complete = False
  
    with st.form(key='tutorial_form'):
        st.image('decision_tree_avatar.svg', width=400)
        st.write(f'Hello {st.session_state.username}, my name is DT.')
        st.write('In each round, you will see data from a customer\'s online shopping session. You will then predict if the customer will made a purchase.')
        st.write('Here is an example:')
        st.dataframe(df.head(1))

        with st.expander('Tip 1'):
            st.write('alkdghalkshdalsh')

        with st.expander('Tip 2'):
            st.write('alkdghalkshdalsh')

        with st.expander('Tip 3'):
            st.write('alkdghalkshdalsh')
        
        tutorial_complete_button = st.form_submit_button(label="Enter game")


        if tutorial_complete_button:
            st.session_state.tutorial_complete = True
        
        st.stop()

  # Show the current question
  if not st.session_state.quiz_complete:
      question_num = st.session_state.question_num
      correct_answer, model_predictions_dict = get_new_question_return_answer(question_df, question_num)
      
      if st.session_state.user_start_time is None:
        st.session_state.user_start_time = time.perf_counter()

      with st.form(key=f'user_prediction_form_{question_num}'):
          user_prediction = st.selectbox(
              label='Predict if the customer will buy a product',
              options=[True, False]
          )
          submit_button = st.form_submit_button(label='Submit')
      if submit_button:

            user_duration = timedelta(seconds=time.perf_counter()-st.session_state.user_start_time)
            st.session_state.user_start_time = None

            if user_prediction == correct_answer:
                st.session_state.user_results[question_num] = True
                #user_results_dict[question_num] = True
                st.success('Correct!')

            if user_prediction != correct_answer:
                st.session_state.user_results[question_num] = False
              #user_results[question_num] = False
                st.error(f'Incorrect. The correct answer is {correct_answer}')
            
            for model, pred in model_predictions_dict.items():
                # In session_state, create an empty list for each model

                if model not in st.session_state.model_results:
                    st.session_state.model_results[model] = []

                #st.session_state.model_results[model][st.session_state.question_num] = (pred == correct_answer)
                #st.write(st.session_state.model_results[model])
                if pred == correct_answer:
                    st.session_state.model_results[model].append(True) 
                    #st.write(st.session_state.model_results[model])
                    #st.header('MODEL SUCCESS')

                elif pred != correct_answer:
                    st.session_state.model_results[model].append(False)
                    #st.write(st.session_state.model_results[model])
                    #st.header('FAIL')
                
                time.sleep(0.5)
            
            st.write(user_duration)

          # Increment the question number
            st.session_state.question_num += 1    
            st.session_state.user_duration = None
      
          # Check if the quiz is complete
            if st.session_state.question_num == total_questions:
                st.session_state.quiz_complete = True
            
            st.header('Generating next question...')
            time.sleep(2)
            st.rerun()


  # Show the completion message
  if st.session_state.quiz_complete:
        user_num_correct = 0
        model_num_correct = 0
        for i in range(total_questions):
            user_question_result = st.session_state.user_results[i]
            if user_question_result == True:
                user_num_correct += 1
        '''
        if user_num_correct > model_num_correct:
            st.balloons()
            st.header("Great job, you beat the model!")
            
        elif user_num_correct < model_num_correct:
            st.header("Womp womp, you failed to beat the model.")
        '''
        # Calculate model scores for each model
        model_scores = {}
        for model, results in st.session_state.model_results.items():
            model_scores[model] = sum(results)

        st.subheader(f'Your score: {user_num_correct} out of {total_questions}')
        st.write(st.session_state.user_results)

        for model, score in model_scores.items():
            st.subheader(f'{model.capitalize()} score: {score} out of {total_questions}')
