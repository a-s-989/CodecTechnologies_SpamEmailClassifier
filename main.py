from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import train_model, evaluate_model, predict_sample
from src.utils import plot_confusion_matrix

def main():
    # Load and preprocess dataset
    df = load_data('data/spam.csv')
    X, y, vectorizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)

    # Select Email sample from csv file
    custom_email = " Lunch at 1 PM works for me."
    
    # Predict and print result
    prediction = predict_sample(model, vectorizer, [custom_email])[0]
    print(f"\nSample Email: {custom_email}")
    print("Prediction:", "Spam" if prediction == 1 else "Not spam")

if __name__ == "__main__":
    main()
