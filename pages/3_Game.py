# Streamlit webapp

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import timedelta

import os
print("Current Directory:", os.getcwd())
print("File Exists:", os.path.exists('ada.pickle'))

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

class SessionState:
    def __init__(self):
        self.username = None
        self.user_results = {}
        self.model_results = {}
        self.quiz_complete = False
        self.tutorial_complete = False
        self.question_num = 0
        self.user_start_time = None
        self.user_duration = {}
    
    def reset(self):
        self.username = ''
        self.user_results = {}
        self.model_results = {}
        self.quiz_complete = False
        self.tutorial_complete = False
        self.question_num = 0
        self.user_start_time = None
        self.user_duration = {}

st.set_page_config(page_title = "Game", 
                   page_icon = "📚",
                   layout= "wide")
#if 'tutorial_complete' in st.session_state and st.session_state.tutorial_complete == False:
#   st.title('Tutorial')

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Predict if a customer will make a purchase</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Compare your results with machine learning models</h3>", unsafe_allow_html=True)

# Load dataset and models
df = load_and_cache_csv('online_shoppers_intention.csv')

ada_pickle = open('ada.pickle', 'rb') 
ada_clf = pickle.load(ada_pickle)
ada_pickle.close()

dt_pickle = open('dt.pickle', 'rb') 
dt_clf = pickle.load(dt_pickle)
dt_pickle.close()

rnd_forest_pickle = open('random_forest.pickle', 'rb') 
rnd_forest_clf = pickle.load(rnd_forest_pickle)
rnd_forest_pickle.close()

#ada_clf = load_and_cache_pickle('model_pickles/ada.pickle')
#dt_clf = load_and_cache_pickle('model_pickles/dt.pickle')
#rnd_forest_clf = load_and_cache_pickle('model_pickles/random_forest.pickle')

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
    
        return correct_answer


models = {'AdaBoost': ada_clf, 'Decision Tree': dt_clf, 'Random Forest': rnd_forest_clf}

def get_model_predictions(models, X, question_num):
    row_encoded = X.iloc[[question_num]]
    model_prediction_dict = {}
    for i, j in models.items():
        start_time = time.perf_counter()
        prediction = j.predict(row_encoded)[0]
        duration = timedelta(seconds=time.perf_counter()-start_time)
        
        model_prediction_dict[i] = [prediction, duration]
        #model_prediction_dict[i + ' duration'] = duration
    
    return model_prediction_dict


total_questions = 7

if 'session_state' not in st.session_state:
    st.session_state['session_state'] = SessionState()

session = st.session_state['session_state']

if session.username == None:

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
    st.markdown('<p class="big-font">Username</p>', unsafe_allow_html=True)
    username_input = st.text_input("", key="username_input")
    username_submit_button = st.form_submit_button(label="Submit")
    
  if username_submit_button:
        if username_input.strip():
                session.username = username_input.strip()
                st.success(f"Welcome, {session.username}!")
                time.sleep(1)  # Brief pause to show the success message
                st.rerun()  # Re-run the app to update the state
        else:
            st.error("Username cannot be empty. Please try again.") 
       
elif session.username != None:
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

  #session.user_results['username'] = session.username
    if not session.tutorial_complete:
        with st.form(key='tutorial_form'):
            st.image('dt_avatar.svg', width=400)
            st.subheader(f'Hello {session.username}, my name is DT...')
            st.write('')
            st.write('- Each round,you will predict if the customer will made a purchase (True) or not (False).')
            st.write('- To prevent information overload, you will only see features with the highest importance. You can see more info about these features on the sidebar.')
            st.write('- Your score is determined by correct predictions and the time taken to answer the question.')
            st.write('')
            st.subheader('Example Row')
            example_row = df.iloc[76]

            # Create a DataFrame from the selected row for display
            example_row_df = example_row[[
                'PageValues',
                'ProductRelated_Duration',
                'BounceRates',
                'ExitRates',
                'Administrative_Duration',
                'ProductRelated'
            ]].to_frame().T  # Convert to DataFrame and transpose to display as a row

            # Display the DataFrame in Streamlit
            st.dataframe(example_row_df, hide_index=True, width=1500)             
            
            st.subheader('Game Tips')

            with st.expander('Tip 1'):
                st.write('PageValues is the most important feature. Higher values are correlated with a shopper making a purchase.')
                st.write(f'Values range from {round(df['PageValues'].min(), 2)} to {round(df['PageValues'].max(), 2)}')

            with st.expander('Tip 2'):
                st.write('ProductRelated_Duration is the 2nd most important feature. Higher values are typically correlated with a shopper making a purchase.')
                st.write(f'Values range from {round(df['ProductRelated_Duration'].min(), 2)} to {round(df['ProductRelated_Duration'].max(), 2)}')
            
            with st.expander('Tip 3'):
                st.write('BounceRates is the 3rd most important feature. Lower values indicate that a shopper is more likely to make a purchase.')
                st.write(f'Values range from {round(df['BounceRates'].min(), 2)} to {round(df['BounceRates'].max(), 2)}')

            tutorial_complete_button = st.form_submit_button(label="Enter game")

            if tutorial_complete_button:
                session.tutorial_complete = True
                st.rerun()

            st.stop()

  # Show the current question
    if not session.quiz_complete:
      correct_answer = get_new_question_return_answer(question_df, session.question_num)
      model_predictions = get_model_predictions(models, X, session.question_num)

      if session.user_start_time is None:
        session.user_start_time = time.perf_counter()

      with st.form(key=f'user_prediction_form_{session.question_num}'):
          user_prediction = st.selectbox(
              label='Predict if the customer will buy a product',
              options=[True, False]
          )
          submit_button = st.form_submit_button(label='Submit')
      if submit_button:

            user_duration = timedelta(seconds=time.perf_counter()-session.user_start_time)
            session.user_start_time = None

            if not session.user_results:
                session.user_results = {'Correct': [], 'Duration': []}

            if user_prediction == correct_answer:
                session.user_results['Correct'].append(True) 
                session.user_results['Duration'].append(user_duration)  
              
                st.success('Correct!')

            if user_prediction != correct_answer:
                session.user_results['Correct'].append(False) 
                session.user_results['Duration'].append(user_duration)
                
                st.error(f'Incorrect. The correct answer is {correct_answer}')
                        
            for model, value in model_predictions.items():
                # In session_state, create an empty list for each model

                if model not in session.model_results:
                    session.model_results[model] = {'Correct': [], 'Duration': []}

                #st.session_state.model_results[model][st.session_state.question_num] = (pred == correct_answer)
                #st.write(st.session_state.model_results[model])
                
                if value[0] == correct_answer:
                    session.model_results[model]['Correct'].append(True) 
                    session.model_results[model]['Duration'].append(value[1])
                    #st.write(st.session_state.model_results[model])
                    #st.header('MODEL SUCCESS')

                elif value[0] != correct_answer:
                    session.model_results[model]['Correct'].append(False)
                    session.model_results[model]['Duration'].append(value[1])

                    #st.write(st.session_state.model_results[model])
                            

          # Increment the question number
            session.question_num += 1    
            session.user_duration = None
      
          # Check if the quiz is complete
            if session.question_num == total_questions:
                session.quiz_complete = True
            
            #st.header('Generating next question...')
            time.sleep(0.5)
            st.rerun()

  # Show the completion message
    if session.quiz_complete:
        st.balloons()
        
        # User Score
        user_score = sum(correct for correct in session.user_results['Correct'])

        # Total User Duration
        total_user_duration = sum(
            session.user_results['Duration'],  # Summing durations directly
            timedelta()  # Initial value for sum
        )

        # Calculate average duration (total duration divided by total questions)
        user_avg_duration = total_user_duration / total_questions

        # Model Scores and Durations
        model_scores = {model: sum(correct for correct in results['Correct']) for model, results in session.model_results.items()}

        model_durations = {}
        for model, results in session.model_results.items():
            # Total duration for each model
            total_duration = sum(
                results['Duration'],  # Summing durations directly
                timedelta()  # Initial value for sum
            )

            # Average duration
            avg_duration = total_duration / total_questions
            model_durations[model] = {"average_duration": avg_duration}

        st.subheader("")
        st.subheader(f'Your score: {user_score} out of {total_questions}')
        st.subheader(f'Average time to answer: {str(user_avg_duration)[5:]}')
        st.write("")

        tab1, tab2, tab3 = st.tabs(["Decision Tree", "Random Forest", "AdaBoost"])

        with tab1:
            st.write("")
            st.subheader(f'Decision Tree score: {model_scores['Decision Tree']} out of {total_questions}')
            st.subheader(f'Average time to answer: {str(model_durations['Decision Tree']['average_duration'])[6:]}')
            st.write("")
            with st.expander('Feature importance'):
                st.image('DT_FI.png')

        with tab2:
            st.write("")
            st.subheader(f'Random Forest score: {model_scores['Random Forest']} out of {total_questions}')
            st.subheader(f'Average time to answer: {str(model_durations['Random Forest']['average_duration'])[6:]}')
            st.write("")
            with st.expander('Feature importance'):
                st.image('RF_FI.png')
            
        with tab3:
            st.write("")
            st.subheader(f'AdaBoost score: {model_scores['AdaBoost']} out of {total_questions}')
            st.subheader(f'Average time to answer: {str(model_durations['AdaBoost']['average_duration'])[6:]}')
            st.write("")
            with st.expander('Feature importance'):
                st.image('ADA_FI.png')