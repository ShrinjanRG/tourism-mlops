import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")

st.title("Travel Package Purchase Prediction")
st.write("Enter the customer details below to predict if they will purchase the new Wellness Tourism Package.")

# Load the saved model safely
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.subheader("Customer Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        monthly_income = st.number_input("Monthly Income", min_value=0, value=25000)
        duration_pitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=120, value=15)
    
    with col2:
        number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=3)
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

    passport_val = 1 if passport == "Yes" else 0
    own_car_val = 1 if own_car == "Yes" else 0

    if st.button("Predict Package Purchase", type="primary"):
        try:
            # Create a base dataframe with all features expected by the model initialized to 0
            if hasattr(model, "feature_names_in_"):
                input_data = pd.DataFrame(columns=model.feature_names_in_)
                input_data.loc[0] = 0  # Fill default zeros for all columns
                
                # Map user inputs to matching column names if they exist in the model
                mapping = {
                    'Age': age,
                    'MonthlyIncome': monthly_income,
                    'DurationOfPitch': duration_pitch,
                    'NumberOfTrips': number_of_trips,
                    'Passport': passport_val,
                    'OwnCar': own_car_val
                }
                for col, val in mapping.items():
                    if col in input_data.columns:
                        input_data.loc[0, col] = val
            else:
                # Fallback if feature names aren't embedded
                input_data = pd.DataFrame([{
                    'Age': age,
                    'MonthlyIncome': monthly_income,
                    'DurationOfPitch': duration_pitch,
                    'NumberOfTrips': number_of_trips,
                    'Passport': passport_val,
                    'OwnCar': own_car_val
                }])

            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success("Result: The customer is **LIKELY** to purchase the Wellness Package! 🎉")
            else:
                st.info("Result: The customer is **UNLIKELY** to purchase the Wellness Package.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
