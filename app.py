import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 


#Load trained model
model =load_model("churn_prediction_model.h5") 

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)  # Load the scaler
    
with open("label_encoder.pkl",'rb') as file:
    label_encoder = pickle.load(file)       # Load the label encoder
    
with open("onehot_encoder.pkl",'rb') as file:
    onehot = pickle.load(file)  # Load the one-hot encoder
    

#Streamlit App
st.title("Customer Churn Prediction App")

geo = st.selectbox("Geography", onehot.categories_[0])  # Use one-hot encoder to get feature names
gender = st.selectbox('Gender', label_encoder.classes_)  # Use label encoder to get classes
age = st.slider('Age', 18, 90)
bal = st.number_input('Balance')
cred_sc = st.number_input('Credit Score')
salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active = st.selectbox('Is Active Member', [0, 1])

#Input data preparation

input_data = {
    'CreditScore': [cred_sc],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [bal],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [salary],
    'Geography': [geo]  # so that drop('Geography') works later
}
input_data=pd.DataFrame(input_data)

#Label encode gender
input_data['Gender'] = label_encoder.transform(input_data['Gender'])


#OneHot Encode Geography
geo_encoded = onehot.transform([[geo]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.drop('Geography',axis=1), geo_encoded_df], axis=1) 
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input)

pred_prob = prediction[0][0]
st.write(f"Prediction Probability: {pred_prob:.2f}")

# Display prediction result
if pred_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay in the Bank.")
    
