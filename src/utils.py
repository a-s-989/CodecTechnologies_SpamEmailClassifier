"""Utility functions."""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Not spam", "Spam"])
    plt.title("Confusion Matrix")
    plt.show()
