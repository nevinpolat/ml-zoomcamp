from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and DictVectorizer
with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    
    # Ensure all required features are present
    required_features = ['job', 'duration', 'poutcome']
    if not all(feature in client_data for feature in required_features):
        return jsonify({'error': 'Missing features in input data'}), 400
    
    # Transform the client data
    X = dv.transform([client_data])
    
    # Predict the probability
    y_pred = model.predict_proba(X)[0, 1]
    
    result = {
        'subscription_probability': round(y_pred, 3)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
