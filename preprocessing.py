import pandas as pd
import re

# Load the dataset
data = pd.read_csv("sms_spam_collection.csv", sep="\t", header=None, names=["Label", "Message"], encoding="utf-8")

# Step 1: Encode the labels (ham -> 0, spam -> 1)
data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

# Step 2: Clean the messages
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["Cleaned_Message"] = data["Message"].apply(clean_text)

# Display the first 5 rows of the processed dataset
print(data.head())

# Save the preprocessed dataset for further steps
data.to_csv("preprocessed_sms_spam.csv", index=False)
