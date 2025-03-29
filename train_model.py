import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("Loading data...")
# Load the training and test data
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

print("Extracting features...")
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform the training data
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']

# Transform the test data
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

print("Training model...")
# Initialize and train the RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(report)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model and vectorizer
print("Saving model and vectorizer...")
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save evaluation metrics
metrics = {
    'accuracy': accuracy,
    'classification_report': report
}
metrics_path = 'model/metrics.pkl'
with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"Evaluation metrics saved to '{metrics_path}'")

print("Model and vectorizer saved to 'model' directory.")
print("You can now run the web app to test your model!")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model and vectorizer
print("Saving model and vectorizer...")
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved to 'model' directory.")
print("You can now run the web app to test your model!")
