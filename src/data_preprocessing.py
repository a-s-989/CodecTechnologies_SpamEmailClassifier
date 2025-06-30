"""Data loading and preprocessing module."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Transform text data into TF-IDF features."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    return X, y, vectorizer

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
