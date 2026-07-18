import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    """Create a sample fake news dataset for demonstration"""

    # Sample real news headlines
    real_news = [
        "Government announces new infrastructure plan",
        "Scientists discover new species in Amazon rainforest",
        "Company reports record quarterly earnings",
        "New study shows benefits of Mediterranean diet",
        "Weather forecast predicts heavy rainfall tomorrow",
        "President meets with world leaders at summit",
        "Technology company launches new smartphone",
        "Research shows decline in unemployment rates",
        "Hospital opens new wing for cancer treatment",
        "University graduates record number of students",
        "Stock market closes higher after positive economic data",
        "Environmental group launches conservation campaign",
        "Medical breakthrough in Alzheimer's research",
        "City council approves new public transportation system",
        "International agreement signed on climate change",
        "New vaccine shows promising results in trials",
        "Economic growth exceeds expectations this quarter",
        "Wildlife sanctuary expands protected areas",
        "Educational reform bill passes legislature",
        "Space agency announces new mission to Mars"
    ]

    # Sample fake news headlines
    fake_news = [
        "SHOCKING: Aliens land in White House lawn!",
        "BREAKING: Celebrity gives birth to alien baby",
        "You won't believe what this politician did next!",
        "URGENT: Government hiding secret about moon landing",
        "MUST SEE: Video of Bigfoot in downtown area",
        "SECRET: Cure for all diseases discovered but suppressed",
        "INCREDIBLE: Man lives 200 years using this one trick",
        "EXCLUSIVE: Photos of unicorn in national park",
        "SHOCKING TRUTH: Dinosaurs still exist in underground caves",
        "BREAKING: Time travel machine invented by scientist",
        "UNBELIEVABLE: Ocean turns to gold overnight",
        "MUST WATCH: Ghost caught on camera in school",
        "SECRET SOCIETY: Illuminati controls world economy",
        "SHOCKING: Zombie apocalypse begins tomorrow",
        "EXCLUSIVE: Proof that Earth is flat revealed",
        "BREAKING: Superhero saves city from villain",
        "You won't believe: Talking dog wins election",
        "URGENT: Volcano erupts in impossible location",
        "MUST SEE: Mermaid spotted in local lake",
        "SECRET: Government replaces citizens with clones"
    ]

    # Create DataFrame
    data = pd.DataFrame({
        'text': real_news + fake_news,
        'label': [1] * len(real_news) + [0] * len(fake_news)  # 1 = real, 0 = fake
    })

    # Add some article bodies (shortened for demo)
    data['full_text'] = data['text'] + " " + np.random.choice([
        "According to experts, this development marks a significant milestone.",
        "The announcement was made during a press conference today.",
        "This comes after months of research and development.",
        "Local residents have mixed reactions to the news.",
        "The implications of this discovery are far-reaching.",
        "More details are expected to be released in the coming days.",
        "This decision affects thousands of people across the region.",
        "Scientists are calling this a breakthrough in their field.",
        "The company plans to invest heavily in this new initiative.",
        "Environmentalists praise the move as a step in the right direction."
    ], size=len(data))

    return data

def save_dataset(data, filename='fake_news_dataset.csv'):
    """Save dataset to CSV file"""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    dataset = create_sample_dataset()
    save_dataset(dataset)
    print("Sample dataset created successfully!")
    print(f"Shape: {dataset.shape}")
    print(f"Label distribution:\n{dataset['label'].value_counts()}")