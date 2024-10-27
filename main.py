#%%writefile project1_main.py
#from google.colab import userdata
#userdata.get('GROQ_API_KEY')

import streamlit as st
import utils as ut

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI

#import streamlit as st
#import pandas as pd
#import pickle
#import numpy as np
#import os
#from openai import OpenAI

#st.title("Customer Churn Prediction")


client=OpenAI(base_url='https://api.groq.com/openai/v1',
              api_key=os.environ.get('GROQ_API_KEY'))

#client=OpenAI(base_url='https://api.groq.com/openai/v1',
#              api_key=userdata.get('GROQ_API_KEY'))



def load_model(filename):
  with open(filename,'rb') as file:
      return pickle.load(file)

xgboost_model=load_model('xgb_model.pkl')
naive_bayes_model=load_model('nb_model.pkl')
random_forest_model=load_model('rf_model.pkl')
decision_tree_model=load_model('dt_model.pkl')
svm_model=load_model('svm_model.pkl') # with svm_model=SVC(random_state=42, probability=True)
knn_model=load_model('knn_model.pkl')
#voting_classifier_model=load_model('/content/voting_clf.pkl')
#xgboost_SMOTE_model=load_model('xgboost-SMOTE.pkl')
#xgboost_featureEgineered_model=load_model('xgboost-featureEgineered.pkl')

# this is in plus algorithm for tutorial
gb_model=load_model('gb_model.pkl')

def prepare_input(amt,lat, zip, city_pop, longg, gender, unix_time,merch_long,merch_lat):

  input_dict={
      'amt':amt,
      'lat':lat,
      'zip':int(zip),
      'city_pop':int(city_pop),
      'long':longg,
      'gender_M':1 if gender=='M' else 0,
      'gender_F':1 if gender=='F' else 0,
      'unix_time':int(unix_time),
      'merch_long':merch_long,
      'merch_lat':merch_lat
  }
  input_df=pd.DataFrame([input_dict])
  return input_df, input_dict


def make_prediction(input_df, input_dict):
    probabilities={'XGBoost':xgboost_model.predict_proba(input_df)[0][1],
                   'Random Forest':random_forest_model.predict_proba(input_df)[0][1],
                   'K-Nearest Neighbors':knn_model.predict_proba(input_df)[0][1],
                   'Support Vector Machine':svm_model.predict_proba(input_df)[0][1],
                   'Gradient Boosting':gb_model.predict_proba(input_df)[0][1],
                   'Decision Tree':decision_tree_model.predict_proba(input_df)[0][1],
                   'Naive Bays':naive_bayes_model.predict_proba(input_df)[0][1],


    }
    avg_probability=np.mean(list(probabilities.values()))

    st.markdown('## Model Probabilities')
    for model, prob in probabilities.items():
      st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_probability}")

    col1, col2 = st.columns(2)

    with col1:
       fig = ut.create_gauge_chart(avg_probability)
       st.plotly_chart(fig, use_container_width=True)
       st.write(f"The customer has a {avg_probability:.2%} probability of fraud.")

    with col2:
       fig_probs = ut.create_model_probability_chart(probabilities)
       st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability


def explain_prediction(probability, input_dict, first, last):

  prompt=f"""You are expert data scientist at a bank for fraud detection, when you specialized interpreting and
  exploring predictions of machine learning models.
  Your machine learning has predicted that a customer named {first} {last} has a
  {round(probability*100, 1)}% probability of fraud, based on the information provided below.
  Here is the customer's information:
  {input_dict}

  Here are the machinen learning model's top 10 most important features for predicting fraud:

               Feature | Importance
  ---------------------------------------------
                   amt |	0.377321
                   lat |	0.105134
                   zip |	0.100778
              city_pop |	0.094403
                  long |	0.084427
              gender_F |	0.066545
             unix_time |	0.064587
            merch_long |	0.064324
             merch_lat |	0.042482
              gender_M |	0.000000


  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for fraud custumers:
  {df[df['is_fraud']==1].describe()}

  Here are summary statistics for non-fraud custumers:
  {df[df['is_fraud']==0].describe()}

  - If the custumers has over a 40% risk of fraud, generate a 3 sentences
  explanation of why they are at risk of fraud.
  - If the custumers has less than a 40% risk of fraud, generate a 3 sentences
  explanation of why they are might not be at risk of fraud.
  - Your explanation should be based on custumer's information, the summary statistics
  of fraud and non-fraud customers, and the feature importances provided.

  Don't mention the probability churning, or the machine learning model, or say
  anything like 'Based on machine learning model's prediction and top 10 most important
  features', just explain the prediction.
  """
  print("EXPLAIN PROMPT", prompt)

  raw_response=client.chat.completions.create(
      model="llama-3.2-3b-preview",
      messages=[
          {"role":"user",
           "content":prompt
           }])

  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation,first,last):

  prompt=f"""You are a manager at HS Bank. You are responsible to send emails at
  customers from the bank if is a fraud or non-fraud transaction.

  Your noticed a customer named {first}{last} has a
  {round(probability*100, 1)}% probability of fraud.

  Here is the customer's information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk of fraud:
  {explanation}

  Generate an email to the customer based on theri information, asking them to say
  if they are at the risk of fraud.

  Make sure to list out set of reasons if was safe transaction at bank or no, 
  in bullet point format. Don't ever mention the probability of fraud, or
  the machine learning model to the customer.

  """
  print("\n\nEMAIL PROMPT", prompt)

  raw_response=client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[
          {"role":"user",
           "content":prompt
           }])

  return raw_response.choices[0].message.content


st.title("Credit Card Fraud Detection")

df=pd.read_csv('fraudTrain.csv')

customers=[f"{row['first']}-{row['last']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer:",customers)

if selected_customer_option:

  selected_first = selected_customer_option.split("-")[0]

  print("First name",selected_first)

  selected_last=selected_customer_option.split("-")[1]

  print("Last name",selected_last)

  selected_customer = df.loc[(df['first'] == selected_first) & (df['last'] == selected_last)].iloc[0]

  print("Selected Customer",selected_customer)

  col1,col2=st.columns(2)

  with col1:
    amt=st.number_input("Amount (amt)",
                        min_value=1.000000,max_value=11872.210000,
                        value=float(selected_customer['amt']))

    lat=st.number_input("Latitude (lat)",
                          min_value=20.027100	,max_value=65.689900,
                          value=float(selected_customer['lat']))

    zip=st.number_input("Zip code",
                        min_value=1257	,max_value=99783	,
                        value=int(selected_customer['zip']))

    city_pop=st.number_input("City Population",
                        min_value=23	, #max_value=2906700.000000	,
                        value=int(selected_customer['city_pop']))

    longg=st.number_input("Longitute (long)",
                     min_value=-165.672300	,max_value=-67.950300	,
                     value=float(selected_customer['long']))

    gender=st.radio("Gender",["Male","Female"],
                   index=0 if selected_customer['gender']=='M' else 1 )




  with col2:
    unix_time=st.number_input("Unix time",
                        min_value=1325376018	,max_value=1326916316 ,
                        value=int(selected_customer['unix_time']))

    merch_long=st.number_input("Merch longitute (merch_long)",
                     min_value=-166.629875		,max_value=-66.967742	,
                     value=float(selected_customer['merch_long']))

    merch_lat=st.number_input("Merch latitude (merch_lat)",
                          min_value=19.040141	,max_value=66.659242	,
                          value=float(selected_customer['merch_lat']))



input_df,input_dict=prepare_input(amt,lat, zip, city_pop, longg, gender, 
                                  unix_time,merch_long,merch_lat)

avg_probability=make_prediction(input_df,input_dict)

explanation=explain_prediction(avg_probability,input_dict,
                    selected_customer['first'], selected_customer['last'])

st.markdown("===")
st.subheader("Explanation of Prediction")
st.markdown(explanation)


email=generate_email(avg_probability,input_dict,explanation,
                    selected_customer['first'], selected_customer['last'])

st.markdown("===")
st.subheader("Personalized Email")
st.markdown(email)

