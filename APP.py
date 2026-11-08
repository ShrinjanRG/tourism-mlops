# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd

# Loading the saved model
model = pickle.load(open('model/model.pkl', 'rb'))

st.title("Wellness Tourism Package Prediction")
st.write("Enter customer details below to predict if they will purchase the package.")

# Example input fields (update these feature names to match your model's training columns)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=50000)

if st.button("Predict"):
    # Create input dataframe for the model
    input_df = pd.DataFrame([[age, income]], columns=['Age', 'MonthlyIncome'])
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("Prediction: The customer is LIKELY to purchase the wellness package!")
    else:
        st.warning("Prediction: The customer is UNLIKELY to purchase the wellness package.")
