# Clickbait Detection Program

clickbait_words = [
    "shocking", "unbelievable", "secret", "amazing",
    "you won't believe", "must watch", "click here",
    "what happened next", "this will blow your mind"
]

def detect_clickbait(title):
    title_lower = title.lower()
    score = 0

    # Keyword check
    for word in clickbait_words:
        if word in title_lower:
            score += 2

    # Pattern checks
    if title.isupper():
        score += 1
    if "!" in title:
        score += 1
    if "?" in title:
        score += 1

    # Result
    if score >= 3:
        return "❌ Clickbait", score
    else:
        return "✅ Not Clickbait", score


# User input
news = input("Enter news title: ")

result, score = detect_clickbait(news)

print("\nResult:", result)
print("Score:", score)