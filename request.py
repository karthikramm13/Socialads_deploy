import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'User ID':15694829, 'Age':32, 'EstimatedSalary':150000})

print(r.json())