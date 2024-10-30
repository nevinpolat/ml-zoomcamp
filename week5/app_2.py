from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and vectorizer when the app starts
with open('model2.bin', 'rb') as model_file:
    model = pickle.load(model_file)

with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        client = request.get_json()
        
        # Validate input
        if not client:
            return jsonify({"error": "No input data provided"}), 400
        
        # Transform the input data using the dictionary vectorizer
        X = dv.transform([client])
        
        # Make prediction using the logistic regression model
        y_pred_proba = model.predict_proba(X)[0, 1]  # Probability of class '1'
        
        # Return the probability as JSON
        return jsonify({"probability": y_pred_proba})
    
    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Welcome to the Subscription Prediction API! Use the /predict endpoint to get predictions."

if __name__ == '__main__':
    # Run the Flask app (optional, since Gunicorn will handle this)
    app.run(host='0.0.0.0', port=9696, debug=True)

