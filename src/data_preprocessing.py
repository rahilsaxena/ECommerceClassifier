import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess_data(df) :
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    return df
