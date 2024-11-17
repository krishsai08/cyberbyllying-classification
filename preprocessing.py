import pandas as pd
import nltk
import contractions
import spacy
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# Define the directory to save the files
save_dir = "D:/project/"
os.makedirs(save_dir, exist_ok=True)  # Creates the directory if it doesn't exist

# Download necessary NLTK data
nltk.download('stopwords')

# Load the dataset
try:
    df = pd.read_csv('D:/project/comments.csv', encoding='latin-1')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    raise FileNotFoundError("The file 'comments.csv' does not exist. Please check the file path.")

# Ensure the dataset contains the necessary columns
required_columns = ['text', 'classification']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Drop rows with missing or invalid data in text/classification
df = df.dropna(subset=['text', 'classification'])

# Convert 'classification' to string type to avoid errors with .str accessor
df['classification'] = df['classification'].astype(str)
df['classification'] = df['classification'].str.strip()

# Remove rows with empty strings in 'text' or 'classification'
df = df[df['text'].str.strip() != '']
df = df[df['classification'] != '']

# Ensure dataset is not empty after cleanup
if df.empty:
    raise ValueError("Dataset is empty after removing missing/invalid data. Check the input data.")

# Data cleaning and preprocessing
print("Starting preprocessing...")

# Remove non-ASCII characters
df['new_comments'] = df['text'].apply(lambda x: x.encode('latin1', errors='replace').decode('utf-8', errors='replace') if isinstance(x, str) else '')

# Lowercase all comments
df['new_comments'] = df['new_comments'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Optional: Expand contractions (e.g., "isn't" -> "is not")
df['new_comments'] = df['new_comments'].apply(lambda x: contractions.fix(x))

# Optional: Remove punctuation
df['new_comments'] = df['new_comments'].str.replace(r'[^\w\s]', '', regex=True)

# Remove digits
df['new_comments'] = df['new_comments'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

# Optional: Remove stop words
stop_words = set(stopwords.words('english'))
df['new_comments'] = df['new_comments'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Lemmatization using SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("SpaCy model 'en_core_web_sm' is not installed. Install it using 'python -m spacy download en_core_web_sm'.")

def lemmatize(comment):
    doc = nlp(comment)
    return ' '.join([token.lemma_ for token in doc])

df['new_comments'] = df['new_comments'].apply(lambda x: lemmatize(x))

# Debugging: Check some of the preprocessed comments
print("Sample preprocessed comments:")
print(df['new_comments'].head())

# Save preprocessed comments and labels
df.to_csv(os.path.join(save_dir, 'preprocessed_comments.csv'), index=False)
print(f"Preprocessed dataset saved. {df.shape[0]} rows remaining.")

# Split the data into training and testing sets
X = df['new_comments']
y = df['classification']

# Handle empty datasets
if X.empty or y.empty:
    raise ValueError("No valid data left for training and testing after preprocessing.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Debugging: Print the shape of the transformed vectors
print(f"Shape of X_train_vec: {X_train_vec.shape}")
print(f"Shape of X_test_vec: {X_test_vec.shape}")

# Save the vectorizer to a file in the specified directory
with open(os.path.join(save_dir, 'vectorizer.pkl'), 'wb') as file:
    pickle.dump(vectorizer, file)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)
with open(os.path.join(save_dir, 'logistic_regression_model.pkl'), 'wb') as file:
    pickle.dump(lr, file)

# Train Random Forest model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_vec, y_train)
with open(os.path.join(save_dir, 'random_forest_model.pkl'), 'wb') as file:
    pickle.dump(rfc, file)

print("Preprocessing and model training complete. Models and vectorizer saved.")
