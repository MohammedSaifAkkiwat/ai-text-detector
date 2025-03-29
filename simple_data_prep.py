from faker import Faker
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

fake = Faker()


def generate_sample_data(num_samples=1000):
    print("Generating sample dataset with Faker...")
    data = []

    # Generate human-like texts using Faker
    for _ in range(num_samples // 2):
        # Generate paragraphs with varying numbers of sentences
        num_sentences = np.random.randint(2, 6)
        text = fake.paragraph(nb_sentences=num_sentences, ext_word_list=None)
        data.append({'text': text, 'label': 'human'})

    # Generate AI-like texts (still using simpler patterns for demonstration)
    for i in range(num_samples // 2):
        # More consistent structure
        num_sentences = 4
        sentence_length = np.random.randint(8, 12)
        words = ['word'] * sentence_length  # Repetitive word structure
        text = '. '.join([' '.join(words).capitalize()
                         for _ in range(num_sentences)]) + '.'
        data.append({'text': text, 'label': 'ai'})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    df.to_csv('data/labeled_data.csv', index=False)
    print(f"Generated {len(df)} samples and saved to data/labeled_data.csv")

    return df


# Generate the data
df = generate_sample_data(2000)

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print("Data split into train and test sets")
print(f"Train set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")
