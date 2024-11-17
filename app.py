from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Flask app instance
app = Flask(__name__)

# Load the saved models and vectorizers
try:
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    
    with open('D:/project/logistic_regression_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    with open('D:/project/random_forest_model.pkl', 'rb') as file:
        rfc_model = pickle.load(file)

except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit(1)  # Exit the program if there's an issue with loading models

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the comment from the form
        comment = request.form['comment'].strip()  # Strip any leading/trailing spaces
        
        if not comment:
            # Handle empty comment case
            return render_template("index.html", error="Please enter a comment.")

        try:
            # Transform the comment using the vectorizer
            comment_vec = vectorizer.transform([comment])
            print(f"Transformed comment vector: {comment_vec.toarray()}")  # Debug: print the transformed feature vector

            # Check if the vector is all zeros (which is problematic)
            if (comment_vec.toarray() == 0).all():
                print("Warning: Comment vector is all zeros. The vectorizer may not be processing this input correctly.")

            # Predict using both models
            prediction_lr = lr_model.predict(comment_vec)
            prediction_rf = rfc_model.predict(comment_vec)

            # Debug: print the raw predictions
            print(f"Logistic Regression Prediction: {prediction_lr[0]}")
            print(f"Random Forest Prediction: {prediction_rf[0]}")

            # Prepare the results to be displayed in a table
            result = {
                "comment": comment,
                "prediction_lr": "Cyberbullying" if prediction_lr[0] == 1 else "Not Cyberbullying",
                "prediction_rf": "Cyberbullying" if prediction_rf[0] == 1 else "Not Cyberbullying"
            }
            return render_template("index.html", result=result)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("index.html", error="Error during prediction. Please try again.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
