import requests
import json

# url = "https://interview-flask-app.herokuapp.com/predict?"
url = "http://127.0.0.1:5001/predict?"
# add our query terms
url += "fixed acidity=7.2&citric acid=0.01&residual sugar=1.7&pH=3.40&sulphates=0.59&alcohol=9.6"

"http://127.0.0.1:5001/predict?fixed+acidity=7.2&citric+acid=0.01&residual+sugar=1.7&pH=3.40&sulphates=0.59&alcohol=9.6
# make the GET request

response = requests.get(url)
# first check the status code
print("status code:", response.status_code)
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses
if response.status_code == 200:
    json_obj = json.loads(response.text)
    print(type(json_obj))
    print(json_obj)