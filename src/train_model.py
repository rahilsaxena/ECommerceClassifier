import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from src.logger import setup_logger
from src.data_preprocessing import preprocess_data

logger = setup_logger("train")

# Load dataset
df = pd.read_csv("data/ecommerce_inquiries.csv")
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
logger.info("Classification Report:\n" + report)
 
# Save
joblib.dump(model, "models/saved_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
logger.info("Model and vectorizer saved successfully.")
