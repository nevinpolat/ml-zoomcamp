import requests

url = 'http://127.0.0.1:9696/predict'

client = {
    "job": "management",
    "duration": 400,
    "poutcome": "success"
}

try:
    response = requests.post(url, json=client)
    response.raise_for_status()  # Raise an error for bad status codes
    result = response.json()
    print(f"Probability of subscription: {result['subscription_probability']}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except KeyError:
    print("Invalid response format.")

