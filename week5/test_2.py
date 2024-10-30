import requests

# Replace with your actual endpoint URL
url = "http://localhost:9696/predict"

# Example client data
client = {
    "job": "management",
    "duration": 400,
    "poutcome": "success"
}

# Make the POST request
response = requests.post(url, json=client)

# Parse and print the response
print(response.json())
