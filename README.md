# Spam Email Classifier

A basic project to classify emails as spam or not spam.

## Features
- Text preprocessing with TF-IDF
- Naive Bayes classifier
- Evaluation metrics and confusion matrix
- Modular code with clear docstrings
- VS Code ready
- CLI arguments for training and predicting

## Installation

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python main.py --train
```

Predict custom email:
```bash
python main.py --predict "Your account has been compromised. Reset your password."
```

## Project Structure

```
.
├── data/
│   └── spam.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── main.py
├── requirements.txt
```

## Future Work
- Add SVM classifier
- Integrate larger real-world dataset
- Create REST API
