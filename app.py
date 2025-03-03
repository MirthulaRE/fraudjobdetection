import streamlit as st
import os
import numpy as np

from tensorflow.keras.models import load_model

# ===========================
# Load the trained model
# ===========================
model_path = "dnn_model.keras"

if os.path.exists(model_path):
    dnn_model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file not found! Please retrain and save `dnn_model.keras`.")
    st.stop()

# ===========================
# Load the TF-IDF Vectorizer
# ===========================
vectorizer_path = "tfidf_vectorizer.pkl"

if os.path.exists(vectorizer_path):
    tfidf_vectorizer = joblib.load(vectorizer_path)
    st.success("✅ TF-IDF Vectorizer loaded successfully!")
else:
    st.error("❌ TF-IDF Vectorizer file not found! Please save `tfidf_vectorizer.pkl`.")
    st.stop()

# ===========================
# Streamlit UI
# ===========================
st.title("🕵️‍♂️ Job Fraud Detection System")

# User input fields
title = st.text_input("Job Title", "")
company_profile = st.text_area("Company Profile", "")
description = st.text_area("Job Description", "")
requirements = st.text_area("Job Requirements", "")
benefits = st.text_area("Job Benefits", "")

# Function to preprocess input
def preprocess_input(title, company_profile, description, requirements, benefits):
    text = f"{title} {company_profile} {description} {requirements} {benefits}"
    return text

# Prediction button
if st.button("🔍 Predict"):
    user_text = preprocess_input(title, company_profile, description, requirements, benefits)
    
    # Convert text using TF-IDF
    user_vectorized = tfidf_vectorizer.transform([user_text]).toarray()

    # Predict using the model
    prediction = dnn_model.predict(user_vectorized)
    fraud_prob = prediction[0][1]  # Probability of being fraudulent
    fraud_prob_percentage = fraud_prob * 100 
    
    # Display result
    if fraud_prob_percentage > 50:
        st.error(f"🚨 This job posting is likely **Fraudulent** .")
    else:
        st.success(f"✅ This job posting is likely **Real** .")
