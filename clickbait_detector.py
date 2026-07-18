import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class ClickbaitDetector:
    def __init__(self):
        # Clickbait keywords and phrases
        self.clickbait_keywords = {
            'high': ['shocking', 'breaking', 'urgent', 'secret', 'exclusive', 'unbelievable',
                    'incredible', 'amazing', 'mind-blowing', 'jaw-dropping', 'heartbreaking',
                    'devastating', 'terrifying', 'horrifying', 'scandalous', 'controversial'],
            'medium': ['you wont believe', 'you will never guess', 'must see', 'must watch',
                      'dont miss', 'this changes everything', 'what happened next',
                      'the reason why', 'the truth about', 'they dont want you to know'],
            'low': ['finally', 'at last', 'now', 'today', 'immediately', 'right now']
        }

        # Clickbait patterns
        self.clickbait_patterns = [
            r'!{2,}',  # Multiple exclamation marks
            r'\?{2,}',  # Multiple question marks
            r'[A-Z]{5,}',  # Excessive capitalization
            r'\d{1,2}:\d{2}',  # Time formats like "10:30"
            r'\$\d+',  # Dollar amounts
            r'\d+%',  # Percentages
            r'^\d+\.',  # Numbered lists
        ]

        # Emotional words
        self.emotional_words = [
            'love', 'hate', 'fear', 'anger', 'joy', 'sadness', 'surprise',
            'disgust', 'trust', 'anticipation', 'shocked', 'outraged',
            'devastated', 'thrilled', 'horrified', 'amazed'
        ]

    def detect_clickbait(self, text):
        """
        Detect if text is clickbait
        Returns: (is_clickbait, score, features)
        """
        if not text or not isinstance(text, str):
            return False, 0, {}

        text_lower = text.lower()
        score = 0
        features = {}

        # Keyword analysis
        keyword_score = self._analyze_keywords(text_lower)
        score += keyword_score
        features['keyword_score'] = keyword_score

        # Pattern analysis
        pattern_score = self._analyze_patterns(text)
        score += pattern_score
        features['pattern_score'] = pattern_score

        # Emotional language analysis
        emotional_score = self._analyze_emotional_language(text_lower)
        score += emotional_score
        features['emotional_score'] = emotional_score

        # Length and structure analysis
        structure_score = self._analyze_structure(text)
        score += structure_score
        features['structure_score'] = structure_score

        # Punctuation analysis
        punctuation_score = self._analyze_punctuation(text)
        score += punctuation_score
        features['punctuation_score'] = punctuation_score

        # Determine if clickbait based on threshold
        is_clickbait = score >= 3  # Adjustable threshold

        features['total_score'] = score
        features['is_clickbait'] = is_clickbait

        return is_clickbait, score, features

    def _analyze_keywords(self, text_lower):
        """Analyze clickbait keywords"""
        score = 0

        for weight, keywords in self.clickbait_keywords.items():
            weight_value = {'high': 2, 'medium': 1, 'low': 0.5}[weight]
            for keyword in keywords:
                if keyword in text_lower:
                    score += weight_value

        return score

    def _analyze_patterns(self, text):
        """Analyze clickbait patterns"""
        score = 0

        for pattern in self.clickbait_patterns:
            matches = re.findall(pattern, text)
            score += len(matches)

        return score

    def _analyze_emotional_language(self, text_lower):
        """Analyze emotional language"""
        score = 0
        words = word_tokenize(text_lower)

        for word in words:
            if word in self.emotional_words:
                score += 1

        return score

    def _analyze_structure(self, text):
        """Analyze text structure"""
        score = 0

        # Short, sensational headlines
        if len(text) < 100 and any(word in text.lower() for word in ['shocking', 'breaking', 'urgent']):
            score += 1

        # Question format
        if text.strip().endswith('?'):
            score += 0.5

        # Numbered lists or steps
        if re.search(r'\d+\.|\b\d+ steps?\b|\b\d+ ways?\b', text.lower()):
            score += 1

        return score

    def _analyze_punctuation(self, text):
        """Analyze punctuation usage"""
        score = 0

        # Excessive punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')

        if exclamation_count > 2:
            score += 1
        if question_count > 2:
            score += 1

        # Multiple punctuation marks together
        if '!!!' in text or '???' in text or '?!' in text:
            score += 1

        return score

    def explain_clickbait(self, text):
        """Provide explanation for clickbait detection"""
        is_clickbait, score, features = self.detect_clickbait(text)

        explanation = []

        if features['keyword_score'] > 0:
            explanation.append(f"Contains {features['keyword_score']:.1f} clickbait keyword points")

        if features['pattern_score'] > 0:
            explanation.append(f"Contains {features['pattern_score']:.1f} clickbait pattern points")

        if features['emotional_score'] > 0:
            explanation.append(f"Uses {features['emotional_score']:.1f} emotional language points")

        if features['structure_score'] > 0:
            explanation.append(f"Has {features['structure_score']:.1f} structural clickbait points")

        if features['punctuation_score'] > 0:
            explanation.append(f"Uses {features['punctuation_score']:.1f} excessive punctuation points")

        if not explanation:
            explanation.append("No significant clickbait characteristics detected")

        return {
            'is_clickbait': is_clickbait,
            'score': score,
            'explanation': explanation,
            'features': features
        }

# Example usage
if __name__ == "__main__":
    detector = ClickbaitDetector()

    test_headlines = [
        "Government announces new policy on healthcare",
        "SHOCKING: You won't believe what happened next!",
        "BREAKING: Major scientific breakthrough discovered",
        "10 ways to improve your productivity today",
        "URGENT: Action required immediately or face consequences",
        "New study shows interesting results about climate change"
    ]

    for headline in test_headlines:
        result = detector.explain_clickbait(headline)
        print(f"\nHeadline: {headline}")
        print(f"Clickbait: {result['is_clickbait']} (Score: {result['score']:.1f})")
        for exp in result['explanation']:
            print(f"- {exp}")
