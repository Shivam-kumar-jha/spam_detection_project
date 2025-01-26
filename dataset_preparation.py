import pandas as pd

# Load the dataset with encoding specified
data = pd.read_csv("sms_spam_collection.csv", sep="\t", header=None, names=["Label", "Message"], encoding="utf-8")

# Display the first 5 rows to confirm successful loading
print(data.head())

# Display dataset info to ensure it's loaded properly
print(data.info())
