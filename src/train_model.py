import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from data_preprocessing import preprocess_data
from logger import setup_logger
import sys
import os
print("Current Working Directory:", os.getcwd())
sys.path.append("/src/train_model.py")
getLog = setup_logger("train")
# Load dataset


df = pd.read_csv("data/ecommerceDataset.csv")

print(pd.Series({"Memory usage": "{:.2f} MB".format(df.memory_usage().sum()/(1024*1024)), 
                 "Dataset shape": "{}".format(df.shape)}).to_string())

df.columns = ['label', 'text']
df['label'] = df['label'].str.strip()
df = preprocess_data(df)
# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
getLog.info("Classification Report:\n" + report)
# Save
joblib.dump(model, "models/saved_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
getLog.info("Model and vectorizer saved successfully.")
