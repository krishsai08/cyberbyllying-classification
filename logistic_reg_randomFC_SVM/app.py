from flask import Flask, request, jsonify, render_template
import re
import contractions
import spacy
from nltk.corpus import stopwords
from joblib import load
import os

# Initialize Flask app
app = Flask(__name__)

# Load models and vectorizer using joblib
vectorizer = load("D:\\project\\logrr_rfc\\models\\vectorizer.joblib")
lr_model = load("D:\\project\\logrr_rfc\\models\\lr_model.joblib")
rfc_model = load("D:\\project\\logrr_rfc\\models\\rfc_model.joblib")
svm_model = load("D:\\project\\logrr_rfc\\models\\svm_model.joblib")  # Load the SVM model

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower()
    comment = contractions.fix(comment)  # Expands contractions (e.g., "I'm" -> "I am")
    comment = re.sub(r"[^\w\s]", "", comment)  # Remove punctuation
    comment = "".join([c for c in comment if not c.isdigit()])  # Remove digits
    stop_words = set(stopwords.words("english"))  # Use NLTK stopwords
    comment = " ".join([word for word in comment.split() if word not in stop_words])  # Remove stopwords
    
    # Lemmatize using spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(comment)
    comment = " ".join(token.lemma_ for token in doc)  # Lemmatization
    return comment

# Define routes
@app.route("/")
def home():
    return render_template("index.html")  # Optional for web interface

@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "comment" not in request.json:
        return jsonify({"error": "Invalid input. Please send a JSON with a 'comment' field."}), 400

    user_comment = request.json["comment"]
    print(f"Received comment: {user_comment}")  # Debugging line
    
    preprocessed_comment = preprocess_comment(user_comment)
    print(f"Preprocessed comment: {preprocessed_comment}")  # Debugging line
    
    # Transform the preprocessed comment using the vectorizer
    comment_vector = vectorizer.transform([preprocessed_comment])

    # Predict using all models
    lr_prediction = lr_model.predict(comment_vector)
    rfc_prediction = rfc_model.predict(comment_vector)
    svm_prediction = svm_model.predict(comment_vector)

    # Prepare the result
    result = {
        "Logistic Regression": "Cyberbullying" if lr_prediction[0] == 1 else "Not Cyberbullying",
        "Random Forest": "Cyberbullying" if rfc_prediction[0] == 1 else "Not Cyberbullying",
        "SVM": "Cyberbullying" if svm_prediction[0] == 1 else "Not Cyberbullying",
        "Preprocessed Comment": preprocessed_comment,
    }
    
    print(f"Result: {result}")  # Debugging line

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
