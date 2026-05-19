from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/churn_model.pkl")


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"})


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    result = "Churn" if prediction == 1 else "No Churn"

    return jsonify({"prediction": result})


# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
