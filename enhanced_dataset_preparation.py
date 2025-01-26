import pandas as pd
import numpy as np
import random

# Load the preprocessed dataset
data = pd.read_csv("preprocessed_sms_spam.csv")

# Generate simulated metadata
np.random.seed(42)
data['Call_Duration'] = np.random.randint(10, 300, size=len(data))  # Call duration in seconds
data['Call_Frequency'] = np.random.randint(1, 10, size=len(data))   # Number of calls in the last week
data['Geographic_Location'] = random.choices(['Urban', 'Rural'], k=len(data))

# One-hot encode the geographic location
data = pd.get_dummies(data, columns=['Geographic_Location'], drop_first=True)

# Save the enhanced dataset
data.to_csv("enhanced_sms_spam.csv", index=False)

print("Enhanced dataset with metadata created successfully!")
