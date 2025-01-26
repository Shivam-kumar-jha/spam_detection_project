import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the enhanced dataset
data = pd.read_csv("enhanced_sms_spam.csv")

# Handle missing values in Cleaned_Message
data["Cleaned_Message"].fillna("", inplace=True)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(data["Cleaned_Message"]).toarray()

# Extract metadata features
metadata_features = data[['Call_Duration', 'Call_Frequency', 'Geographic_Location_Urban']].values

# Standardize metadata features
scaler = StandardScaler()
metadata_features = scaler.fit_transform(metadata_features)

# Combine text and metadata features
X = np.hstack((X_text, metadata_features))
y = data["Label"].values

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save for model training
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Features combined and saved successfully!")
