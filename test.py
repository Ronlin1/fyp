import requests
import json

# URL of your running Flask API endpoint
url = "http://localhost:5000/predict"

# Data from your CSV (example: first row )
data = {
    "r1h": 14.0,
    "r1h_avg": 3.8,
    "r3h": 6.0,
    "r3h_avg": 5.9,
    "rfq": 0.9,
    "r1q": 0.8
}

# You can also send multiple rows as a list of dictionaries
# data = [
#     {"r1h": 4.0, "r1h_avg": 3.8, "r3h": 6.0, "r3h_avg": 5.9, "rfq": 0.9, "r1q": 0.8},
#     {"r1h": 4.2, "r1h_avg": 4.0, "r3h": 6.2, "r3h_avg": 6.0, "rfq": 0.85, "r1q": 0.82}
# ]

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print("Success:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
