import requests

comment = {'reddit_comment':'Testing a comment.'}

url = "http://localhost:8000/predict"
response = requests.post(url, json=comment)
print(response.json())