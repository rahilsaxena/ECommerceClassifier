import joblib
from src.logger import setup_logger

logger = setup_logger("inference")

model = joblib.load("models/saved_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def classify_text(text):
    from src.data_preprocessing import clean_text
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    logger.info(f"Input: {text} | Predicted: {prediction}")
    return prediction 