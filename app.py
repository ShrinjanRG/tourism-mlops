import streamlit as st
import pickle
import os

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        st.error("model.pkl not found in the root directory!")
        return None

model = load_model()
