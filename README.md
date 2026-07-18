# Fake News Detection with Clickbait Analysis

A comprehensive software application for detecting fake news and identifying clickbait content using advanced natural language processing and machine learning techniques.

## Features

### 📰 Fake News Detection
- Utilizes Naive Bayes and Logistic Regression classifiers
- Trained on diverse news datasets
- Provides confidence scores for predictions

### 🎣 Clickbait Detection
- Advanced keyword and pattern recognition
- Analyzes punctuation, capitalization, and emotional language
- Detailed scoring system with explanations

### 🔍 Explanation System
- Highlights suspicious words and patterns
- Provides reasoning for each prediction
- Shows probability distributions

### 📊 Similar News Comparison
- Compares input with related articles
- Verifies consistency and authenticity
- Uses TF-IDF vectorization for similarity matching

### 📈 Statistics Dashboard
- Visualizes analysis results
- Shows fake vs real news distribution
- Tracks user activity and detection trends
- Interactive charts and metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
```bash
python src/data_preparation.py
```

4. Train the model:
```bash
python src/train_model.py
```

5. Run the application:
```bash
streamlit run src/app.py
```

## Project Structure

```
fake-news-detection/
├── data/                    # Dataset files
├── models/                  # Trained models and vectorizers
├── src/                     # Source code
│   ├── app.py              # Main Streamlit application
│   ├── train_model.py      # Model training script
│   ├── data_preparation.py # Data preparation utilities
│   ├── clickbait_detector.py # Clickbait detection module
│   └── similar_news_finder.py # Similar news comparison
├── .venv/                   # Virtual environment
└── README.md               # Project documentation
```

## Usage

1. **Detection Page**: Enter news headlines or articles to analyze
2. **Dashboard Page**: View statistics and analysis history
3. **About Page**: Learn more about the application

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **BeautifulSoup**: Web scraping capabilities

## Model Performance

The system uses a Naive Bayes classifier trained on a balanced dataset of real and fake news articles, achieving high accuracy in distinguishing between authentic and misleading content.

## Future Enhancements

- Real-time news API integration
- Multilingual support
- Advanced deep learning models (BERT, LSTM)
- User feedback system for model improvement
- Mobile application development

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes. Always verify information from multiple reliable sources before making decisions based on news content.
