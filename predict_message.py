import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
with open("enhanced_spam_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the dataset for vectorizer and scaler fitting
data = pd.read_csv("enhanced_sms_spam.csv")
data["Cleaned_Message"] = data["Cleaned_Message"].fillna("")

# Initialize the vectorizer with the same parameters as in training
vectorizer = TfidfVectorizer(max_features=3000)  # Match the training dimension
vectorizer.fit(data["Cleaned_Message"])

# Initialize the scaler with the same metadata features as in training
scaler = StandardScaler()
scaler.fit(data[["Call_Duration", "Call_Frequency", "Geographic_Location_Urban"]])

# Function to predict spam or ham
def predict_message(message, call_duration, call_frequency, location):
    # Vectorize the message
    text_features = vectorizer.transform([message]).toarray()
    # Prepare metadata
    metadata_features = np.array([[call_duration, call_frequency, 1 if location == "Urban" else 0]])
    metadata_features = scaler.transform(metadata_features)
    # Combine text and metadata features
    features = np.hstack((text_features, metadata_features))
    # Make prediction
    prediction = model.predict(features)[0]
    return "Spam" if prediction == 1 else "Ham"

# User input for testing
print("Enter details for spam detection:")
message = input("Message: ")
call_duration = int(input("Call Duration (in seconds): "))
call_frequency = int(input("Call Frequency (calls in the past week): "))
location = input("Geographic Location (Urban/Rural): ")

# Predict and display the result
result = predict_message(message, call_duration, call_frequency, location)
print(f"The message is classified as: {result}")
