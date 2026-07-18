import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os

# Import custom modules
from clickbait_detector import ClickbaitDetector
from similar_news_finder import SimilarNewsFinder

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize detectors
clickbait_detector = ClickbaitDetector()
similar_finder = SimilarNewsFinder()

# Load or train model
@st.cache_resource
def load_model():
    model_path = 'models/fake_news_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    else:
        # Train model if not exists
        return train_model()

def train_model():
    # Load sample data (in real implementation, use larger dataset)
    data = pd.DataFrame({
        'text': [
            "Breaking: Scientists discover cure for cancer",
            "Local man wins lottery for third time",
            "Government announces new tax policy",
            "Celebrity caught in scandal",
            "Stock market crashes due to economic downturn",
            "New study shows benefits of exercise",
            "Politician makes controversial statement",
            "Weather forecast predicts sunny weekend",
            "Company announces record profits",
            "Social media trend goes viral",
            "You won't believe what happened next!",
            "SHOCKING: This changes everything!",
            "MUST SEE: Incredible footage revealed",
            "URGENT: Action required immediately",
            "SECRET: Hidden truth exposed"
        ],
        'label': [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # 1 = real, 0 = fake
    })

    # Preprocess text
    data['processed_text'] = data['text'].apply(preprocess_text)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['processed_text'])
    y = data['label']

    # Train model
    model = MultinomialNB()
    model.fit(X, y)

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def detect_clickbait(text):
    """Use the ClickbaitDetector class"""
    result = clickbait_detector.explain_clickbait(text)
    return result['is_clickbait'], result['score'], result['explanation']

def explain_prediction(text, model, vectorizer):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vectorized)[0]
    else:
        probs = [0.5, 0.5]  # Default for models without predict_proba

    # Simple explanation based on keywords
    fake_keywords = ['breaking', 'shocking', 'secret', 'urgent', 'scandal']
    real_keywords = ['study', 'announces', 'forecast', 'company', 'government']

    suspicious_words = []
    for word in fake_keywords:
        if word in text.lower():
            suspicious_words.append(word)

    return suspicious_words, probs

def find_similar_news(text):
    """Use the SimilarNewsFinder class"""
    similar_articles = similar_finder.find_similar_news_mock(text, 5)
    return [article['title'] for article in similar_articles]

def main():
    st.title("📰 Fake News Detection with Clickbait Analysis")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Detection", "Dashboard", "About"])

    if page == "Detection":
        detection_page()
    elif page == "Dashboard":
        dashboard_page()
    else:
        about_page()

def detection_page():
    st.header("News Analysis")

    # Input
    news_text = st.text_area("Enter news headline or article:", height=100)

    if st.button("Analyze"):
        if news_text.strip():
            model, vectorizer = load_model()

            # Preprocess
            processed = preprocess_text(news_text)
            vectorized = vectorizer.transform([processed])

            # Predict
            prediction = model.predict(vectorized)[0]
            confidence = max(model.predict_proba(vectorized)[0]) if hasattr(model, 'predict_proba') else 0.5

            # Clickbait detection
            is_clickbait, clickbait_score, clickbait_explanation = detect_clickbait(news_text)

            # Explanation
            suspicious_words, probs = explain_prediction(news_text, model, vectorizer)

            # Similar news
            similar_news = find_similar_news(news_text)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.success(f"✅ Real News (Confidence: {confidence:.2f})")
                else:
                    st.error(f"❌ Fake News (Confidence: {confidence:.2f})")

                if is_clickbait:
                    st.warning(f"⚠️ Clickbait Detected (Score: {clickbait_score})")
                    with st.expander("Clickbait Details"):
                        for exp in clickbait_explanation:
                            st.write(f"• {exp}")
                else:
                    st.info("✅ No Clickbait Detected")

            with col2:
                st.subheader("Explanation")
                if suspicious_words:
                    st.write("Suspicious words/patterns detected:")
                    for word in suspicious_words:
                        st.write(f"- {word}")
                else:
                    st.write("No obvious suspicious patterns detected.")

                st.write(f"Probability distribution: Real: {probs[1]:.2f}, Fake: {probs[0]:.2f}")

            st.subheader("Similar News Articles")
            for news in similar_news:
                st.write(f"• {news}")

            # Add to history
            st.session_state.history.append({
                'text': news_text[:100] + '...' if len(news_text) > 100 else news_text,
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': confidence,
                'clickbait': is_clickbait,
                'timestamp': pd.Timestamp.now()
            })

        else:
            st.warning("Please enter some text to analyze.")

def dashboard_page():
    st.header("📊 Statistics Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)

        with col1:
            total_analyzed = len(df)
            st.metric("Total Analyzed", total_analyzed)

        with col2:
            fake_count = (df['prediction'] == 'Fake').sum()
            st.metric("Fake News Detected", fake_count)

        with col3:
            clickbait_count = df['clickbait'].sum()
            st.metric("Clickbait Detected", clickbait_count)

        # Charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Prediction distribution
        prediction_counts = df['prediction'].value_counts()
        ax1.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%')
        ax1.set_title('Prediction Distribution')

        # Confidence histogram
        ax2.hist(df['confidence'], bins=10, edgecolor='black')
        ax2.set_title('Confidence Scores')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')

        st.pyplot(fig)

        # Recent history
        st.subheader("Recent Analysis History")
        st.dataframe(df.tail(10)[['text', 'prediction', 'confidence', 'clickbait']])

    else:
        st.info("No analysis history yet. Start by analyzing some news!")

def about_page():
    st.header("About")
    st.write("""
    This application detects fake news and clickbait using advanced NLP and machine learning techniques.

    **Features:**
    - Fake news detection using Naive Bayes classifier
    - Clickbait analysis with keyword and pattern recognition
    - Explanation system highlighting suspicious content
    - Similar news comparison
    - Statistics dashboard with visualizations
    - Confidence scoring for predictions

    **Technologies Used:**
    - Python
    - Scikit-learn
    - NLTK
    - Streamlit
    - Pandas, NumPy
    """)

if __name__ == "__main__":
    main()