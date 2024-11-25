from flask import Flask, request, jsonify, render_template
import re
import contractions
import spacy
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define model and tokenizer paths
MODEL_PATH_RNN = "D:\\project\\RNN_LSTM_models\\models\\rnn_model.h5"
MODEL_PATH_LSTM = "D:\\project\\RNN_LSTM_models\\models\\lstm_model.h5"
TOKENIZER_PATH = "D:\\project\\RNN_LSTM_models\\models\\tokenizer.pkl"

'''MODEL_PATH_RNN = "D:\\project\\RNN_LSTM_models\\models\\rnn_model (1).h5"
MODEL_PATH_LSTM = "D:\\project\\RNN_LSTM_models\\models\\lstm_model (1).h5"
TOKENIZER_PATH = "D:\\project\\RNN_LSTM_models\\models\\tokenizer (1).pkl"   '''


# Load the RNN and LSTM models and tokenizer
try:
    rnn_model = load_model(MODEL_PATH_RNN)
    lstm_model = load_model(MODEL_PATH_LSTM)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Load spaCy model once to avoid repeated loading
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"Error loading spaCy model: {e}")

# Stopwords for preprocessing
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_comment(comment):
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Expand contractions
        comment = contractions.fix(comment)

        # Remove punctuation and digits
        comment = re.sub(r"[^\w\s]", "", comment)  # Remove punctuation
        comment = "".join([c for c in comment if not c.isdigit()])  # Remove digits

        # Remove stopwords
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatize with spaCy
        doc = nlp(comment)
        comment = " ".join(token.lemma_ for token in doc)

        return comment

    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

# Define routes
@app.route("/")
def home():
    # Optional route to serve a front-end template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.json or "comment" not in request.json:
            return jsonify({"error": "Invalid input. Please send a JSON with a 'comment' field."}), 400

        user_comment = request.json["comment"]
        print(f"[INFO] Received comment: {user_comment}")  # Debugging line

        # Preprocess the comment
        preprocessed_comment = preprocess_comment(user_comment)
        print(f"[INFO] Preprocessed comment: {preprocessed_comment}")  # Debugging line

        # Tokenize and pad the comment
        comment_sequence = tokenizer.texts_to_sequences([preprocessed_comment])
        padded_comment = pad_sequences(comment_sequence, maxlen=100, padding='pre')
        print(f"[INFO] Padded comment: {padded_comment}")  # Debugging line

        # Predict using both RNN and LSTM models
        rnn_prediction = rnn_model.predict(padded_comment)
        lstm_prediction = lstm_model.predict(padded_comment)
        
        print(f"[INFO] RNN model prediction: {rnn_prediction}")  # Debugging line
        print(f"[INFO] LSTM model prediction: {lstm_prediction}")  # Debugging line

        # Average the probabilities from both models
        avg_prediction = (rnn_prediction[0][0] + lstm_prediction[0][0]) / 2
        print(f"[INFO] Average prediction: {avg_prediction}")  # Debugging line

        # Convert the averaged probability to a class (0 or 1) based on threshold
        final_prediction = 1 if avg_prediction >= 0.5 else 0

        # Determine the final prediction label
        prediction_label = "Cyberbullying" if final_prediction == 1 else "Not Cyberbullying"

        # Prepare and return the result
        result = {
            "Classification": prediction_label
        }
        print(f"[INFO] Result: {result}")  # Debugging line

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    app.run(debug=True)
