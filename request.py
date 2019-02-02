import requests

url = 'http://localhost:5000/api'

# if request.method == 'POST':
# 		message = request.form['message']
# 		data = [message]

r = requests.post(url,json={'exp': "This is obscene"})
print(r.json())