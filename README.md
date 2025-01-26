# Spam Detection Project

This project focuses on developing a **Spam Detection System** tailored for financial security challenges. The solution integrates text analysis, metadata processing, and advanced machine learning techniques to detect spam messages effectively.

## Features
- **Spam Classification**: Identifies whether a message is spam or not.
- **Metadata Integration**: Uses call duration, frequency, and geographic location to enhance predictions.
- **Streamlit Web App**: Provides a user-friendly interface for testing the system.
- **Oversampling with SMOTE**: Balances the dataset to improve recall for spam detection.

---

## Directory Structure
```
spam_detection_project/
├── spam_detection_app.py          # Streamlit app for testing the system
├── feature_extraction.py          # Prepares features for training
├── model_training.py              # Trains the spam detection model
├── predict_message.py             # Predicts spam or ham for new inputs
├── enhanced_sms_spam.csv          # Preprocessed dataset with metadata
├── enhanced_spam_detection_model.pkl # Trained machine learning model
├── requirements.txt               # Dependencies for the project
└── README.md                      # Project documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or later
- Virtual environment setup (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd spam_detection_project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate   # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run spam_detection_app.py
   ```

---

## Usage
1. Open the Streamlit app in your browser (usually `http://localhost:8501`).
2. Input the message, call duration, frequency, and location.
3. Click the **Predict** button to classify the message as `Spam` or `Ham`.

---

## Technical Details

### 1. Preprocessing
- Cleans messages by removing special characters.
- Encodes metadata (e.g., geographic location).

### 2. Feature Engineering
- **TfidfVectorizer**: Converts text into numerical vectors (3000 features).
- **StandardScaler**: Standardizes metadata (call duration, frequency, location).

### 3. Model Training
- **Logistic Regression**: Trained using `SGDClassifier` for batch processing.
- **SMOTE**: Balances the dataset by oversampling the minority class (spam).

### 4. Deployment
- Built using **Streamlit** for a web-based interface.

---

## Results
| Metric         | Value   |
|----------------|---------|
| Accuracy       | 96.95%  |
| Precision (Spam)| 100%    |
| Recall (Spam)  | 77%     |
| F1-Score (Spam)| 87%     |

---

## Future Enhancements
1. Use advanced models like **XGBoost** or **Transformers**.
2. Add real-time feedback to improve the model dynamically.
3. Deploy on platforms like AWS or Heroku for broader access.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- **UCI SMS Spam Collection Dataset** for the dataset.
- **Scikit-Learn** and **Streamlit** for the tools and libraries.