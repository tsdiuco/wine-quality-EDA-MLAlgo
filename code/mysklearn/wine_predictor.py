import requests
import json

# url = "https://interview-flask-app.herokuapp.com/predict?"
url = "http://127.0.0.1:5001/predict?"
# add our query terms
url += "residul+sugar=1.5&alcohol=9.7&cirtic+acid=0.02&fixed+acidity=7.1"

# make the GET request

response = requests.get(url)
# first check the status code
print("status code:", response.status_code)
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses
if response.status_code == 200:
    json_obj = json.loads(response.text)
    print(type(json_obj))
    print(json_obj)