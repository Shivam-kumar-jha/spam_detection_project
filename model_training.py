import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import pickle

# Load the enhanced features
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Apply SMOTE to oversample the minority class (spam)
smote = SMOTE(random_state=42)

# Break the dataset into chunks for SMOTE to handle memory efficiently
chunk_size = 2000
X_train_chunks = np.array_split(X_train, len(X_train) // chunk_size + 1)
y_train_chunks = np.array_split(y_train, len(y_train) // chunk_size + 1)

X_resampled_list = []
y_resampled_list = []

# Apply SMOTE on each chunk
for X_chunk, y_chunk in zip(X_train_chunks, y_train_chunks):
    X_res, y_res = smote.fit_resample(X_chunk, y_chunk)
    X_resampled_list.append(X_res)
    y_resampled_list.append(y_res)

# Concatenate all resampled chunks
X_train_resampled = np.vstack(X_resampled_list)
y_train_resampled = np.hstack(y_resampled_list)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Shuffle the data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Initialize the SGDClassifier model
model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)

# Train the model in batches using partial_fit
batch_size = 1000
classes = np.unique(y_train)  # Define classes for partial_fit

for i in range(0, len(X_train_resampled), batch_size):
    batch_X = X_train_resampled[i:i + batch_size]
    batch_y = y_train_resampled[i:i + batch_size]
    model.partial_fit(batch_X, batch_y, classes=classes)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Enhanced Model Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open("enhanced_spam_detection_model.pkl", "wb") as file:
    pickle.dump(model, file)
