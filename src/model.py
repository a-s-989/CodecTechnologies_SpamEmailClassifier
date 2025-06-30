"""Model training and evaluation module."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def train_model(X_train, y_train):
    """Train Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def predict_sample(model, vectorizer, texts):
    """Predict label for new texts."""
    X_new = vectorizer.transform(texts)
    return model.predict(X_new)
