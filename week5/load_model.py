import pickle

# Load DictVectorizer and LogisticRegression model
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

# Client data
client = {"job": "management", "duration": 400, "poutcome": "success"}

# Transform client data
X = dv.transform([client])

# Predict probability
proba = model.predict_proba(X)[0][1]
print(proba)
