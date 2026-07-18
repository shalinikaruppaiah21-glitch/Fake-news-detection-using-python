import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SimilarNewsFinder:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def preprocess_text(self, text):
        """Preprocess text for similarity comparison"""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

    def find_similar_news_mock(self, query_text, num_results=5):
        """Mock similar news finder for demonstration"""
        # Mock news database
        mock_news = [
            "Government announces new infrastructure development plan",
            "Scientists make breakthrough in renewable energy research",
            "Company reports strong financial results for Q4",
            "New study reveals benefits of healthy eating habits",
            "Weather forecast predicts sunny conditions for weekend",
            "President meets with international leaders at summit",
            "Technology firm launches innovative new product",
            "Research shows positive trends in job market",
            "Hospital expands services with new medical wing",
            "University celebrates record graduation numbers",
            "Stock market shows gains after economic report",
            "Environmental organization starts conservation initiative",
            "Medical researchers announce Alzheimer's treatment progress",
            "City council approves public transportation improvements",
            "Countries sign historic climate change agreement",
            "New vaccine demonstrates effectiveness in clinical trials",
            "Economy grows faster than expected this quarter",
            "Wildlife preserve increases protected land area",
            "Education reform legislation passes through parliament",
            "Space agency plans ambitious Mars exploration mission"
        ]

        # Preprocess all texts
        processed_query = self.preprocess_text(query_text)
        processed_news = [self.preprocess_text(news) for news in mock_news]

        # Vectorize
        all_texts = [processed_query] + processed_news
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

            # Get top similar news
            top_indices = np.argsort(similarities)[::-1][:num_results]
            similar_articles = []

            for idx in top_indices:
                if similarities[idx] > 0.1:  # Similarity threshold
                    similar_articles.append({
                        'title': mock_news[idx],
                        'similarity': similarities[idx],
                        'source': 'Mock News Database'
                    })

            return similar_articles

        except:
            # Fallback if vectorization fails
            return self._fallback_similar_news(query_text, num_results)

    def _fallback_similar_news(self, query_text, num_results):
        """Fallback method using keyword matching"""
        mock_news = [
            "Government announces new infrastructure development plan",
            "Scientists make breakthrough in renewable energy research",
            "Company reports strong financial results for Q4",
            "New study reveals benefits of healthy eating habits",
            "Weather forecast predicts sunny conditions for weekend"
        ]

        query_words = set(self.preprocess_text(query_text).split())
        similar_articles = []

        for news in mock_news:
            news_words = set(self.preprocess_text(news).split())
            common_words = query_words.intersection(news_words)
            similarity = len(common_words) / len(query_words) if query_words else 0

            if similarity > 0.1:
                similar_articles.append({
                    'title': news,
                    'similarity': similarity,
                    'source': 'Fallback Search'
                })

        return sorted(similar_articles, key=lambda x: x['similarity'], reverse=True)[:num_results]

    def find_similar_news_web(self, query_text, num_results=5):
        """
        Find similar news using web search (requires API key for real implementation)
        This is a mock implementation
        """
        # In a real implementation, you would use:
        # - Google News API
        # - Bing News API
        # - NewsAPI.org
        # - Web scraping from news websites

        # For now, return mock results
        return self.find_similar_news_mock(query_text, num_results)

    def compare_with_similar(self, original_text, similar_articles):
        """Compare original text with similar articles for consistency check"""
        if not similar_articles:
            return "No similar articles found for comparison"

        high_similarity_count = sum(1 for article in similar_articles if article['similarity'] > 0.5)

        if high_similarity_count > 0:
            return f"Found {high_similarity_count} highly similar articles. Content appears consistent with existing news."
        else:
            return "Limited similarity with existing articles. May contain novel or potentially false information."

# Example usage
if __name__ == "__main__":
    finder = SimilarNewsFinder()

    test_queries = [
        "Government announces new policy",
        "SHOCKING: Aliens discovered on Mars",
        "Company reports record profits"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        similar = finder.find_similar_news_mock(query, 3)
        for article in similar:
            print(".3f")
        comparison = finder.compare_with_similar(query, similar)
        print(f"Consistency check: {comparison}")