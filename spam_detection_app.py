import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load model and data
with open("enhanced_spam_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

data = pd.read_csv("enhanced_sms_spam.csv")
data["Cleaned_Message"] = data["Cleaned_Message"].fillna("")

# Initialize vectorizer and scaler
vectorizer = TfidfVectorizer(max_features=3000)
vectorizer.fit(data["Cleaned_Message"])

scaler = StandardScaler()
scaler.fit(data[["Call_Duration", "Call_Frequency", "Geographic_Location_Urban"]])

# Streamlit interface
st.title("Spam Call Detection System")
st.write("Enter details to classify the message as Spam or Ham.")

# Inputs
message = st.text_area("Enter the message:", "")
call_duration = st.number_input("Call Duration (in seconds):", min_value=0)
call_frequency = st.number_input("Call Frequency (calls in the past week):", min_value=0)
location = st.selectbox("Geographic Location:", ["Urban", "Rural"])

if st.button("Predict"):
    # Prepare input features
    text_features = vectorizer.transform([message]).toarray()
    metadata_features = np.array([[call_duration, call_frequency, 1 if location == "Urban" else 0]])
    metadata_features = scaler.transform(metadata_features)
    features = np.hstack((text_features, metadata_features))

    # Make prediction
    prediction = model.predict(features)[0]
    result = "Spam" if prediction == 1 else "Ham"
    st.success(f"The message is classified as: {result}")
