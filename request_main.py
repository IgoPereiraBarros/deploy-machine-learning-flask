# https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c

import requests

url = 'http://localhost:5000/api'

r = requests.post(url, json={'exp': 1.7})

print(r.json())
