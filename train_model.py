import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text for model training"""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def load_data(filepath='data/fake_news_dataset.csv'):
    """Load and preprocess the dataset"""
    if not os.path.exists(filepath):
        print(f"Dataset file {filepath} not found. Running data preparation...")
        from data_preparation import create_sample_dataset, save_dataset
        data = create_sample_dataset()
        save_dataset(data)
    else:
        data = pd.read_csv(filepath)

    # Preprocess text
    data['processed_text'] = data['text'].apply(preprocess_text)

    return data

def train_models(X_train, X_test, y_train, y_test, vectorizer):
    """Train and evaluate different models"""
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    return results

def save_model(model, vectorizer, model_name='naive_bayes'):
    """Save trained model and vectorizer"""
    os.makedirs('models', exist_ok=True)

    model_path = f'models/{model_name}_model.pkl'
    vectorizer_path = f'models/{model_name}_vectorizer.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()

def main():
    print("Loading data...")
    data = load_data()

    print(f"Dataset shape: {data.shape}")
    print(f"Label distribution:\n{data['label'].value_counts()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'], data['label'],
        test_size=0.2, random_state=42, stratify=data['label']
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training data shape: {X_train_vec.shape}")
    print(f"Test data shape: {X_test_vec.shape}")

    # Train models
    print("\nTraining models...")
    results = train_models(X_train_vec, X_test_vec, y_train, y_test, vectorizer)

    # Save best model (Naive Bayes for simplicity)
    best_model = results['Naive Bayes']['model']
    save_model(best_model, vectorizer, 'fake_news')

    # Plot confusion matrix for best model
    plot_confusion_matrix(y_test, results['Naive Bayes']['predictions'], 'Naive Bayes')

    print("\nModel training completed!")

if __name__ == "__main__":
    main()